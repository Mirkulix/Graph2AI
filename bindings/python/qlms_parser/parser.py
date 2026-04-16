"""
QLMS v1.1 Python Reference Parser
=================================

A minimal, stdlib-only parser + verifier for the QLANG Message Stream
binary envelope. Exists to prove protocol language-independence for the
Linux Foundation AAIF submission (spec §16.4 item 6).

Wire formats supported:

    v1 unsigned (``AgentConversation::to_binary``):
        [QLMS magic : 4 B]
        [count      : u32 LE]
        [JSON payload]

    v2 signed (``AgentConversation::to_signed_binary``):
        [QLMS magic    : 4 B]
        [version       : u16 LE = 2]
        [flags         : u16 LE, bit0 = SIGNED]
        [signature     : 64 B]   = r(32) || s(32)
        [pubkey        : 32 B]
        [payload_hash  : 32 B]   = SHA-256(JSON payload)
        [msg_count     : u32 LE]
        [JSON payload]

Signing scheme (matches ``qlang-core::crypto::Keypair``):

    r        = HMAC-SHA256(secret_key, msg)     # not verifiable publicly
    tag      = SHA-256(r || pubkey || msg)
    s        = SHA-256(tag)
    sig      = r || s                           # 64 bytes

Because ``r`` is unforgeable without the secret and ``s = SHA-256(tag)``
is fully determined by ``(r, pubkey, msg)``, the verifier only needs:

    expected_s = SHA-256(SHA-256(r || pubkey || msg))

and a constant-time byte comparison against ``s``.

Dependencies: Python standard library only (``hashlib``, ``hmac``,
``struct``, ``json``, ``dataclasses``). No third-party crates, no C
extensions. Tested against Python 3.10+.
"""

from __future__ import annotations

import hashlib
import hmac as _hmac_stdlib
import json
import struct
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Constants (spec v1.0 §3 / v1.1 §14)
# ---------------------------------------------------------------------------

MAGIC: bytes = b"QLMS"
"""The 4-byte envelope prefix: ``0x51 0x4C 0x4D 0x53``."""

VERSION_V2: int = 2
"""Wire version for signed envelopes."""

FLAG_SIGNED: int = 0x0001
"""Bit 0 of the flags field: envelope carries a signature + pubkey + hash."""

SIGNATURE_LEN: int = 64
PUBKEY_LEN: int = 32
HASH_LEN: int = 32


class QlmsParseError(ValueError):
    """Raised when bytes are not a well-formed QLMS envelope."""


class QlmsSignatureError(ValueError):
    """Raised when an envelope parses but signature / hash verification fails."""


# ---------------------------------------------------------------------------
# Crypto helpers — stdlib only
# ---------------------------------------------------------------------------


def sha256(data: bytes) -> bytes:
    """Return the 32-byte SHA-256 digest of ``data``."""
    return hashlib.sha256(data).digest()


def ct_eq(a: bytes, b: bytes) -> bool:
    """Constant-time byte comparison.

    Uses ``hmac.compare_digest`` from the standard library — the reference
    timing-safe primitive on CPython. Returns ``False`` on length mismatch
    without leaking which input was shorter.
    """
    if len(a) != len(b):
        return False
    return _hmac_stdlib.compare_digest(a, b)


# ---------------------------------------------------------------------------
# Parsed frame types
# ---------------------------------------------------------------------------


@dataclass
class UnsignedFrame:
    """v1 envelope: magic + count + JSON payload, no auth block."""

    msg_count: int
    payload: bytes

    @property
    def is_signed(self) -> bool:
        return False

    def messages(self) -> list:
        """Return the JSON-decoded messages list."""
        return json.loads(self.payload)


@dataclass
class SignedFrame:
    """v2 envelope: magic + version + flags + sig + pubkey + hash + count + payload."""

    version: int
    flags: int
    signature: bytes  # 64 B
    pubkey: bytes  # 32 B
    payload_hash: bytes  # 32 B
    msg_count: int
    payload: bytes

    @property
    def is_signed(self) -> bool:
        return (self.flags & FLAG_SIGNED) != 0

    def messages(self) -> list:
        return json.loads(self.payload)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _check_magic(buf: bytes) -> None:
    if len(buf) < 4:
        raise QlmsParseError(f"buffer too short for QLMS magic ({len(buf)} B)")
    if buf[:4] != MAGIC:
        raise QlmsParseError(
            f"bad magic: expected {MAGIC!r}, got {buf[:4]!r}"
        )


def parse_v1(buf: bytes) -> UnsignedFrame:
    """Parse an unsigned v1 envelope. Raises :class:`QlmsParseError` on malformed input."""
    _check_magic(buf)
    if len(buf) < 8:
        raise QlmsParseError("v1 envelope needs >= 8 bytes (magic + count)")
    (count,) = struct.unpack_from("<I", buf, 4)
    payload = buf[8:]
    return UnsignedFrame(msg_count=count, payload=payload)


def parse_v2(buf: bytes) -> SignedFrame:
    """Parse a signed v2 envelope. Raises :class:`QlmsParseError` on malformed input."""
    _check_magic(buf)
    # header: magic(4) + version(2) + flags(2) + sig(64) + pubkey(32) + hash(32) + count(4) = 140
    header_len = 4 + 2 + 2 + SIGNATURE_LEN + PUBKEY_LEN + HASH_LEN + 4
    if len(buf) < header_len:
        raise QlmsParseError(
            f"v2 envelope needs >= {header_len} B, got {len(buf)}"
        )

    version, flags = struct.unpack_from("<HH", buf, 4)
    if version != VERSION_V2:
        raise QlmsParseError(f"unsupported v2 version field: {version}")
    if (flags & FLAG_SIGNED) == 0:
        raise QlmsParseError("parse_v2 called on unsigned envelope (SIGNED flag unset)")

    offset = 8
    signature = bytes(buf[offset : offset + SIGNATURE_LEN])
    offset += SIGNATURE_LEN
    pubkey = bytes(buf[offset : offset + PUBKEY_LEN])
    offset += PUBKEY_LEN
    payload_hash = bytes(buf[offset : offset + HASH_LEN])
    offset += HASH_LEN
    (msg_count,) = struct.unpack_from("<I", buf, offset)
    offset += 4
    payload = bytes(buf[offset:])

    return SignedFrame(
        version=version,
        flags=flags,
        signature=signature,
        pubkey=pubkey,
        payload_hash=payload_hash,
        msg_count=msg_count,
        payload=payload,
    )


# ---------------------------------------------------------------------------
# Verification (v2 only)
# ---------------------------------------------------------------------------


def verify_signed_frame(frame: SignedFrame) -> bool:
    """Return True iff the v2 envelope's signature and payload_hash check out.

    Steps:

    1. Recompute ``SHA-256(payload)`` and constant-time compare against
       ``frame.payload_hash``. Mismatch → tampering after signing.
    2. Split ``signature`` into ``r || s``.
    3. Recompute ``tag = SHA-256(r || pubkey || payload_hash)`` and
       ``expected_s = SHA-256(tag)``.
    4. Constant-time compare ``expected_s`` with ``s``.

    Any step's failure returns ``False``. This mirrors
    ``Keypair::verify(pubkey, payload_hash, signature)`` in Rust.
    """
    if not frame.is_signed:
        return False

    # Step 1: payload integrity.
    recomputed_hash = sha256(frame.payload)
    if not ct_eq(recomputed_hash, frame.payload_hash):
        return False

    # Step 2-4: signature verification over the payload hash.
    r = frame.signature[:32]
    s = frame.signature[32:]
    tag = sha256(r + frame.pubkey + frame.payload_hash)
    expected_s = sha256(tag)
    return ct_eq(s, expected_s)


def require_verified(frame: SignedFrame) -> None:
    """Raise :class:`QlmsSignatureError` if the frame does not verify."""
    if not verify_signed_frame(frame):
        raise QlmsSignatureError("QLMS v2 signature verification failed")


# ---------------------------------------------------------------------------
# Pubkey derivation (matches Keypair::from_seed in qlang-core)
# ---------------------------------------------------------------------------


def derive_pubkey(seed: bytes) -> bytes:
    """Reproduce ``Keypair::from_seed(seed).public_key()``.

    Formula: ``pubkey = SHA-256(b"qlang-pubkey" || seed)`` (qlang-core
    `crypto.rs`, lines 229-234). Used by the golden test to confirm that
    the Python side computes the same public key from a known seed.
    """
    if len(seed) != 32:
        raise ValueError("seed must be exactly 32 bytes")
    return sha256(b"qlang-pubkey" + seed)


# ---------------------------------------------------------------------------
# Convenience: dispatch by peeking at the envelope.
# ---------------------------------------------------------------------------


def parse_auto(buf: bytes):
    """Parse either v1 or v2 based on the flags/version bytes after the magic.

    Heuristic: after the 4-byte magic, if the next ``u16`` (little-endian)
    equals ``VERSION_V2`` **and** the subsequent ``u16`` has ``FLAG_SIGNED``
    set, treat as v2. Otherwise fall back to v1.

    Returns :class:`UnsignedFrame` or :class:`SignedFrame`.
    """
    _check_magic(buf)
    if len(buf) >= 8:
        version, flags = struct.unpack_from("<HH", buf, 4)
        if version == VERSION_V2 and (flags & FLAG_SIGNED):
            return parse_v2(buf)
    return parse_v1(buf)
