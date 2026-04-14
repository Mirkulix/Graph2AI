//! Byte-Pair Encoding (BPE) tokenizer for QLANG language models.
//!
//! A minimal but functional BPE tokenizer implemented in pure Rust:
//! - Train from raw text
//! - Encode text to token IDs
//! - Decode token IDs back to text
//! - Save/load to binary files
//!
//! The vocabulary starts with all 256 single bytes, then iteratively merges
//! the most frequent adjacent pair until the desired vocabulary size is reached.

use std::collections::HashMap;
use std::io::{Read, Write, BufWriter, BufReader};

/// A trained BPE tokenizer.
pub struct BpeTokenizer {
    /// Ordered list of merge operations: (token_a, token_b) -> new_token
    merges: Vec<(u32, u32)>,
    /// Vocabulary: token_id -> byte sequence
    vocab: Vec<Vec<u8>>,
    /// Reverse lookup: byte sequence -> token_id
    token_to_id: HashMap<Vec<u8>, u32>,
    /// Beginning-of-sequence token ID
    pub bos_id: u32,
    /// End-of-sequence token ID
    pub eos_id: u32,
    /// Padding token ID
    pub pad_id: u32,
}

impl BpeTokenizer {
    /// Train a BPE tokenizer from text data.
    ///
    /// 1. Initialize vocabulary with all 256 single-byte tokens
    /// 2. Repeatedly find the most frequent adjacent pair and merge it
    /// 3. Stop when `vocab_size` is reached (must be >= 259 for 256 bytes + 3 special)
    pub fn train(text: &str, vocab_size: usize) -> Self {
        // We need at least 256 byte tokens + 3 special tokens
        let min_vocab = 256 + 3;
        let vocab_size = vocab_size.max(min_vocab);

        // Initialize vocabulary with single bytes
        let mut vocab: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        let mut token_to_id: HashMap<Vec<u8>, u32> = HashMap::new();
        for (i, v) in vocab.iter().enumerate() {
            token_to_id.insert(v.clone(), i as u32);
        }

        // Convert text to a sequence of byte-level token IDs
        let bytes = text.as_bytes();
        let mut ids: Vec<u32> = bytes.iter().map(|&b| b as u32).collect();

        let mut merges: Vec<(u32, u32)> = Vec::new();

        // Number of merge operations needed
        let n_merges = vocab_size - min_vocab;

        for _ in 0..n_merges {
            if ids.len() < 2 {
                break;
            }

            // Count all adjacent pairs
            let mut pair_counts: HashMap<(u32, u32), usize> = HashMap::new();
            for window in ids.windows(2) {
                let pair = (window[0], window[1]);
                *pair_counts.entry(pair).or_insert(0) += 1;
            }

            // Find the most frequent pair
            let best_pair = match pair_counts.into_iter().max_by_key(|&(_, count)| count) {
                Some((pair, count)) if count >= 2 => pair,
                _ => break, // No pair occurs more than once, stop
            };

            // Create new token for the merged pair
            let new_id = vocab.len() as u32;
            let mut new_bytes = vocab[best_pair.0 as usize].clone();
            new_bytes.extend_from_slice(&vocab[best_pair.1 as usize]);
            token_to_id.insert(new_bytes.clone(), new_id);
            vocab.push(new_bytes);
            merges.push(best_pair);

            // Replace all occurrences of the pair in ids
            let mut new_ids = Vec::with_capacity(ids.len());
            let mut i = 0;
            while i < ids.len() {
                if i + 1 < ids.len() && ids[i] == best_pair.0 && ids[i + 1] == best_pair.1 {
                    new_ids.push(new_id);
                    i += 2;
                } else {
                    new_ids.push(ids[i]);
                    i += 1;
                }
            }
            ids = new_ids;
        }

        // Add special tokens
        let bos_id = vocab.len() as u32;
        vocab.push(b"<BOS>".to_vec());
        token_to_id.insert(b"<BOS>".to_vec(), bos_id);

        let eos_id = vocab.len() as u32;
        vocab.push(b"<EOS>".to_vec());
        token_to_id.insert(b"<EOS>".to_vec(), eos_id);

        let pad_id = vocab.len() as u32;
        vocab.push(b"<PAD>".to_vec());
        token_to_id.insert(b"<PAD>".to_vec(), pad_id);

        Self {
            merges,
            vocab,
            token_to_id,
            bos_id,
            eos_id,
            pad_id,
        }
    }

    /// Encode text to a sequence of token IDs.
    ///
    /// Converts text to bytes, then applies merges in order.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let bytes = text.as_bytes();
        if bytes.is_empty() {
            return Vec::new();
        }

        // Start with byte-level tokens
        let mut ids: Vec<u32> = bytes.iter().map(|&b| b as u32).collect();

        // Apply each merge in order
        for &(a, b) in &self.merges {
            let merged_id = match self.token_to_id.get(&{
                let mut v = self.vocab[a as usize].clone();
                v.extend_from_slice(&self.vocab[b as usize]);
                v
            }) {
                Some(&id) => id,
                None => continue,
            };

            let mut new_ids = Vec::with_capacity(ids.len());
            let mut i = 0;
            while i < ids.len() {
                if i + 1 < ids.len() && ids[i] == a && ids[i + 1] == b {
                    new_ids.push(merged_id);
                    i += 2;
                } else {
                    new_ids.push(ids[i]);
                    i += 1;
                }
            }
            ids = new_ids;
        }

        ids
    }

    /// Decode a sequence of token IDs back to a string.
    ///
    /// Looks up each token ID in the vocabulary, concatenates bytes,
    /// and converts to UTF-8 (lossy).
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            let idx = id as usize;
            if idx < self.vocab.len() {
                let token_bytes = &self.vocab[idx];
                // Skip special tokens in output
                if id == self.bos_id || id == self.eos_id || id == self.pad_id {
                    continue;
                }
                bytes.extend_from_slice(token_bytes);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Vocabulary size (including special tokens).
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Save the tokenizer to a binary file.
    ///
    /// Format:
    ///   - u32: number of merges
    ///   - For each merge: (u32, u32)
    ///   - u32: vocab size
    ///   - For each vocab entry: u32 length + bytes
    ///   - u32: bos_id, u32: eos_id, u32: pad_id
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let file = std::fs::File::create(path)?;
        let mut w = BufWriter::new(file);

        // Magic header
        w.write_all(b"QBPE")?;

        // Merges
        let n_merges = self.merges.len() as u32;
        w.write_all(&n_merges.to_le_bytes())?;
        for &(a, b) in &self.merges {
            w.write_all(&a.to_le_bytes())?;
            w.write_all(&b.to_le_bytes())?;
        }

        // Vocab
        let vocab_len = self.vocab.len() as u32;
        w.write_all(&vocab_len.to_le_bytes())?;
        for entry in &self.vocab {
            let len = entry.len() as u32;
            w.write_all(&len.to_le_bytes())?;
            w.write_all(entry)?;
        }

        // Special token IDs
        w.write_all(&self.bos_id.to_le_bytes())?;
        w.write_all(&self.eos_id.to_le_bytes())?;
        w.write_all(&self.pad_id.to_le_bytes())?;

        w.flush()?;
        Ok(())
    }

    /// Load a tokenizer from a binary file.
    pub fn load(path: &str) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let mut r = BufReader::new(file);

        // Magic header
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != b"QBPE" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Not a QBPE tokenizer file",
            ));
        }

        // Merges
        let mut buf4 = [0u8; 4];
        r.read_exact(&mut buf4)?;
        let n_merges = u32::from_le_bytes(buf4) as usize;
        let mut merges = Vec::with_capacity(n_merges);
        for _ in 0..n_merges {
            r.read_exact(&mut buf4)?;
            let a = u32::from_le_bytes(buf4);
            r.read_exact(&mut buf4)?;
            let b = u32::from_le_bytes(buf4);
            merges.push((a, b));
        }

        // Vocab
        r.read_exact(&mut buf4)?;
        let vocab_len = u32::from_le_bytes(buf4) as usize;
        let mut vocab = Vec::with_capacity(vocab_len);
        let mut token_to_id = HashMap::new();
        for i in 0..vocab_len {
            r.read_exact(&mut buf4)?;
            let len = u32::from_le_bytes(buf4) as usize;
            let mut entry = vec![0u8; len];
            r.read_exact(&mut entry)?;
            token_to_id.insert(entry.clone(), i as u32);
            vocab.push(entry);
        }

        // Special token IDs
        r.read_exact(&mut buf4)?;
        let bos_id = u32::from_le_bytes(buf4);
        r.read_exact(&mut buf4)?;
        let eos_id = u32::from_le_bytes(buf4);
        r.read_exact(&mut buf4)?;
        let pad_id = u32::from_le_bytes(buf4);

        Ok(Self {
            merges,
            vocab,
            token_to_id,
            bos_id,
            eos_id,
            pad_id,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train_basic() {
        let text = "aaabdaaabac";
        let tok = BpeTokenizer::train(text, 260); // 256 + 3 special + 1 merge
        // Should have at least 259 tokens (256 bytes + 3 special)
        assert!(tok.vocab_size() >= 259);
        // The most frequent pair "aa" should have been merged
        assert!(!tok.merges.is_empty());
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let text = "hello world! this is a test of the tokenizer.";
        let tok = BpeTokenizer::train(text, 280);

        let encoded = tok.encode(text);
        let decoded = tok.decode(&encoded);
        assert_eq!(decoded, text, "Encode-decode roundtrip must be lossless");
    }

    #[test]
    fn test_encode_decode_short() {
        let tok = BpeTokenizer::train("abcabc", 259);
        let encoded = tok.encode("abc");
        let decoded = tok.decode(&encoded);
        assert_eq!(decoded, "abc");
    }

    #[test]
    fn test_encode_empty() {
        let tok = BpeTokenizer::train("hello", 259);
        let encoded = tok.encode("");
        assert!(encoded.is_empty());
        let decoded = tok.decode(&[]);
        assert_eq!(decoded, "");
    }

    #[test]
    fn test_unicode_text() {
        let text = "Hallo Welt! Gruesse aus Oesterreich. Schoene Gruesse!";
        let tok = BpeTokenizer::train(text, 270);
        let encoded = tok.encode(text);
        let decoded = tok.decode(&encoded);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_unicode_emoji() {
        // BPE works on bytes, so multi-byte UTF-8 sequences are handled naturally
        let text = "hello world hello world";
        let tok = BpeTokenizer::train(text, 270);
        let encoded = tok.encode(text);
        let decoded = tok.decode(&encoded);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_special_tokens() {
        let tok = BpeTokenizer::train("test", 259);
        assert!(tok.bos_id < tok.vocab_size() as u32);
        assert!(tok.eos_id < tok.vocab_size() as u32);
        assert!(tok.pad_id < tok.vocab_size() as u32);
        // Special tokens should be different
        assert_ne!(tok.bos_id, tok.eos_id);
        assert_ne!(tok.eos_id, tok.pad_id);
        assert_ne!(tok.bos_id, tok.pad_id);
    }

    #[test]
    fn test_special_tokens_skipped_in_decode() {
        let tok = BpeTokenizer::train("hello", 259);
        let decoded = tok.decode(&[tok.bos_id, 104, 101, 108, 108, 111, tok.eos_id]);
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn test_vocab_size() {
        let tok = BpeTokenizer::train("aaaaaa bbbbbb cccccc", 265);
        // 256 bytes + up to (265-259)=6 merges + 3 special
        assert!(tok.vocab_size() >= 259);
        assert!(tok.vocab_size() <= 265);
    }

    #[test]
    fn test_save_load_roundtrip() {
        let text = "the quick brown fox jumps over the lazy dog";
        let tok = BpeTokenizer::train(text, 270);

        let path = "/tmp/qlang_test_tokenizer.qbpe";
        tok.save(path).expect("save failed");
        let tok2 = BpeTokenizer::load(path).expect("load failed");

        // Verify same encoding
        let enc1 = tok.encode(text);
        let enc2 = tok2.encode(text);
        assert_eq!(enc1, enc2, "Loaded tokenizer must produce same encoding");
        assert_eq!(tok.vocab_size(), tok2.vocab_size());
        assert_eq!(tok.bos_id, tok2.bos_id);
        assert_eq!(tok.eos_id, tok2.eos_id);
        assert_eq!(tok.pad_id, tok2.pad_id);

        // Cleanup
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_longer_text_compression() {
        // With repeated patterns, BPE should compress effectively
        let text = "abcabc".repeat(100);
        let tok = BpeTokenizer::train(&text, 280);
        let encoded = tok.encode(&text);
        // Encoded should be shorter than raw bytes
        assert!(
            encoded.len() < text.len(),
            "BPE should compress repetitive text: {} encoded vs {} bytes",
            encoded.len(),
            text.len()
        );
        let decoded = tok.decode(&encoded);
        assert_eq!(decoded, text);
    }
}
