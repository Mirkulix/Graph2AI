# Crypto

Alle Kryptographie in QLANG ist in reinem Rust implementiert -- keine externen Crates. Siehe [[Decisions]] fuer das Warum.

## SHA-256

Vollstaendige SHA-256 Implementierung in `qlang-core/crypto.rs` (~60 Zeilen):

```rust
use qlang_core::crypto::sha256;

let hash: [u8; 32] = sha256(b"Hello QLANG");
```

Verwendet fuer:
- Graph Content Hashing im [[BinaryFormat]]
- [[Protocol]] Payload Hashing
- Merkle Tree Knoten-Hashes
- Content-Addressable Cache Keys

## HMAC-SHA256

Hash-based Message Authentication Code:

```rust
use qlang_core::crypto::hmac_sha256;

let mac: [u8; 32] = hmac_sha256(key, message);
```

Verwendet fuer:
- Signieren von `GraphMessage` im [[Protocol]]
- Verifikation der Nachrichtenintegritaet

## Keypair

Kryptographische Schluesselpaar-Generierung:

```rust
use qlang_core::crypto::Keypair;

let keypair = Keypair::generate();
// keypair.public_key: [u8; 32]
// keypair.signing_key: [u8; 32]
```

## Signierte Graphen

Jeder Graph kann kryptographisch signiert werden:

### Signing Flow

1. SHA-256 Hash des Graphen berechnen
2. Hash mit Keypair signieren (HMAC-SHA256, Ed25519-kompatibles Wire Format)
3. Signatur (64 bytes), Public Key (32 bytes) und Hash (32 bytes) an Nachricht anhaengen

### Verifikation

1. SHA-256 des aktuellen Graphen neu berechnen
2. Pruefen ob er mit dem gespeicherten Hash uebereinstimmt (erkennt Manipulation)
3. Signatur gegen Public Key verifizieren

```
GraphMessage {
    signature: Option<[u8; 64]>,    // HMAC-SHA256 Signatur
    signer_pubkey: Option<[u8; 32]>, // Oeffentlicher Schluessel
    graph_hash: Option<[u8; 32]>,   // SHA-256 zum Zeitpunkt der Signierung
}
```

## Merkle Trees

Merkle-Baum ueber Graph-Knoten fuer partielle Verifikation (`merkle.rs`):

```
           Root Hash
          /         \
     Hash(0,1)    Hash(2,3)
      /    \       /    \
  Node0  Node1  Node2  Node3
```

### Wie es funktioniert

Jeder Knoten bekommt einen SHA-256 Hash basierend auf:
- Operation (Op Display)
- Input-Typen
- Output-Typen

Diese Blatt-Hashes bilden einen binaeren Baum bis zur Wurzel.

### MerkleProof

```rust
pub struct MerkleProof {
    pub node_id: u32,                    // Fuer welchen Knoten
    pub node_hash: [u8; 32],             // SHA-256 des Knotens
    pub siblings: Vec<([u8; 32], bool)>, // Geschwister-Hashes + Position
    pub root: [u8; 32],                  // Erwarteter Wurzel-Hash
}
```

### Anwendungsfaelle

| Anwendung | Beschreibung |
|-----------|-------------|
| Partielle Verifikation | Beweisen, dass ein einzelner Knoten zu einem signierten Graphen gehoert |
| Inkrementelle Updates | Bei Graph-Aenderung nur den geaenderten Pfad neu hashen |
| Verteiltes Vertrauen | Beweise teilen ohne den vollstaendigen Graphen zu teilen |

## Content-Addressable Cache

Der Computation Cache (`cache.rs`) verwendet SHA-256 fuer inhaltsbasierte Schluessel:

```
Cache Key = SHA-256(graph_hash || input_hash_0 || input_hash_1 || ...)
```

### ComputationCache

```rust
use qlang_core::cache::ComputationCache;

let cache = ComputationCache::global();

// Lookup
if let Some(entry) = cache.lock().unwrap().get(&key) {
    // Cache hit -- verwende gecachte Ergebnisse
    return entry.outputs;
}

// Miss -- berechnen und cachen
cache.lock().unwrap().put(key, outputs, compute_time_us);
```

### Features
- **LRU Eviction**: Aelteste Eintraege werden entfernt wenn max_entries (default: 10.000) erreicht
- **Hit/Miss Statistiken**: `cache stats` zeigt Trefferquote
- **Thread-safe**: Global Mutex-geschuetzt
- **Timing**: Speichert wie lange die originale Berechnung dauerte

### CLI

```bash
qlang-cli cache stats   # Hit/Miss Statistiken anzeigen
qlang-cli cache clear   # Cache leeren
```

## HTTP-zu-QLMS Signing Proxy

Der Signing Proxy (`qlang-cli proxy`) signiert automatisch alle durchgehenden Graphen:

```bash
qlang-cli proxy --port 9100 --upstream http://localhost:8081
```

- Empfaengt HTTP-Requests
- Signiert den Graph-Payload mit dem konfigurierten Keypair
- Leitet an Upstream weiter
- Transparent fuer den Client

## Sicherheitsphilosophie

1. **Keine externen Crypto-Crates** -- jede Zeile ist auditierbar
2. **Supply Chain Security** -- keine transitiven Abhaengigkeiten koennen Code injizieren
3. **Jedes Byte auf dem Wire verstehen** -- kritisch fuer ein Sicherheitsprotokoll
4. **Einfach genug fuer ein Nachmittags-Audit** -- SHA-256 ~60 Zeilen, WebSocket ~100 Zeilen

Siehe [[Protocol]] fuer das Wire Format, [[BinaryFormat]] fuer die Graph-Serialisierung.

#crypto #sha256 #merkle #security #signing
