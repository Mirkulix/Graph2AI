"""
QLMS v1.1 Golden-File Test
==========================

Reads the binary fixtures emitted by
``cargo run -p qlang-agent --example emit_golden`` and asserts the Python
reference parser agrees with the Rust reference implementation. Exit code
0 = all fixtures passed; non-zero = at least one fixture failed.

Fixtures live in ``./fixtures/`` as pairs:

    fixtures/<name>.qlms   # binary envelope
    fixtures/<name>.json   # metadata (seed, expected pubkey, flags, ...)

This script is the "prove language-independence" artefact referenced by
spec §16.4 item 6 and the AAIF submission checklist.

Usage:

    python golden_test.py                     # run all fixtures
    python golden_test.py v2_signed_single    # run a single fixture
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Local import — parser.py sits next to this file.
from parser import (
    FLAG_SIGNED,
    QlmsParseError,
    QlmsSignatureError,
    derive_pubkey,
    parse_auto,
    parse_v1,
    parse_v2,
    require_verified,
    verify_signed_frame,
)

FIXTURES_DIR: Path = Path(__file__).parent / "fixtures"


class TestFailure(AssertionError):
    """Raised when a golden fixture does not match the Python parser's view."""


# ---------------------------------------------------------------------------
# Per-fixture assertions
# ---------------------------------------------------------------------------


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise TestFailure(msg)


def _load(name: str):
    """Load the ``(binary_bytes, metadata_dict)`` tuple for a fixture."""
    bin_path = FIXTURES_DIR / f"{name}.qlms"
    meta_path = FIXTURES_DIR / f"{name}.json"
    if not bin_path.exists():
        raise FileNotFoundError(f"missing binary fixture: {bin_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"missing metadata fixture: {meta_path}")
    data = bin_path.read_bytes()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return data, meta


def check_v1_unsigned(name: str, data: bytes, meta: dict) -> None:
    frame = parse_v1(data)
    _assert(
        frame.msg_count == meta["msg_count"],
        f"[{name}] msg_count mismatch: python={frame.msg_count} "
        f"rust={meta['msg_count']}",
    )
    # JSON payload must decode and have the expected number of messages.
    msgs = frame.messages()
    _assert(
        isinstance(msgs, list) and len(msgs) == meta["msg_count"],
        f"[{name}] payload JSON must decode to a list of length {meta['msg_count']}",
    )
    # parse_auto should route this to v1 (no SIGNED flag).
    auto = parse_auto(data)
    _assert(
        not auto.is_signed,
        f"[{name}] parse_auto must not treat v1 as signed",
    )


def check_v2_signed(name: str, data: bytes, meta: dict) -> None:
    frame = parse_v2(data)

    _assert(
        frame.version == meta["version"],
        f"[{name}] version mismatch: python={frame.version} rust={meta['version']}",
    )
    _assert(
        (frame.flags & FLAG_SIGNED) != 0,
        f"[{name}] SIGNED flag must be set (flags={hex(frame.flags)})",
    )
    _assert(
        frame.msg_count == meta["msg_count"],
        f"[{name}] msg_count mismatch: python={frame.msg_count} rust={meta['msg_count']}",
    )

    # Pubkey derivation matches Rust's Keypair::from_seed(seed).
    seed_hex = meta["seed_hex"]
    seed = bytes.fromhex(seed_hex)
    expected_pub = derive_pubkey(seed)
    _assert(
        expected_pub.hex() == meta["expected_pubkey_hex"],
        f"[{name}] Python-derived pubkey does not match Rust's: "
        f"derived={expected_pub.hex()} meta={meta['expected_pubkey_hex']}",
    )
    _assert(
        frame.pubkey.hex() == meta["expected_pubkey_hex"],
        f"[{name}] envelope pubkey does not match expected: "
        f"envelope={frame.pubkey.hex()} meta={meta['expected_pubkey_hex']}",
    )

    # Signature must verify using Python's recomputation.
    _assert(
        verify_signed_frame(frame),
        f"[{name}] Python verify_signed_frame returned False on a valid fixture",
    )
    # require_verified must NOT raise.
    require_verified(frame)

    # Tampering detection — mutate the payload and ensure verify now fails.
    import dataclasses

    tampered = dataclasses.replace(
        frame,
        payload=frame.payload + b" ",  # whitespace → SHA-256 changes
    )
    _assert(
        not verify_signed_frame(tampered),
        f"[{name}] verify_signed_frame must reject a payload-tampered frame",
    )
    try:
        require_verified(tampered)
    except QlmsSignatureError:
        pass
    else:
        raise TestFailure(
            f"[{name}] require_verified must raise QlmsSignatureError on tamper"
        )

    # parse_auto should pick v2.
    auto = parse_auto(data)
    _assert(
        auto.is_signed,
        f"[{name}] parse_auto must detect SIGNED flag",
    )


# ---------------------------------------------------------------------------
# Fixture registry
# ---------------------------------------------------------------------------

CHECKERS = {
    "v1_unsigned": check_v1_unsigned,
    "v2_signed": check_v2_signed,
}


def run_one(name: str) -> bool:
    """Run a single fixture. Returns True on success, False on failure."""
    try:
        data, meta = _load(name)
    except FileNotFoundError as exc:
        print(f"  ! {name}: {exc}")
        return False

    kind = meta.get("kind")
    checker = CHECKERS.get(kind)
    if checker is None:
        print(f"  ! {name}: unknown 'kind' in metadata: {kind!r}")
        return False

    try:
        checker(name, data, meta)
    except (TestFailure, QlmsParseError, QlmsSignatureError) as exc:
        print(f"  FAIL {name}  ({kind})\n      {exc}")
        return False

    print(f"  PASS {name}  ({kind}, {len(data)} B)")
    return True


def discover() -> list[str]:
    if not FIXTURES_DIR.is_dir():
        return []
    names = sorted(
        {p.stem for p in FIXTURES_DIR.glob("*.qlms")}
        & {p.stem for p in FIXTURES_DIR.glob("*.json")}
    )
    return names


def main(argv: list[str]) -> int:
    selected = argv[1:] if len(argv) > 1 else discover()
    if not selected:
        print(
            "No fixtures found. Run the Rust emitter first:\n"
            "    cargo run -p qlang-agent --example emit_golden"
        )
        return 2

    print(f"Running {len(selected)} QLMS golden fixture(s) from {FIXTURES_DIR}")
    failed = 0
    for name in selected:
        if not run_one(name):
            failed += 1

    total = len(selected)
    passed = total - failed
    print(f"\n{passed}/{total} fixtures passed", end="")
    if failed:
        print(f", {failed} failed.")
        return 1
    print(".")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
