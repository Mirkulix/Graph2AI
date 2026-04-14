//! MNIST Data Loader — Load the real MNIST dataset.
//!
//! Downloads and parses the IDX binary format:
//! - train-images-idx3-ubyte (60000 images, 28×28)
//! - train-labels-idx1-ubyte (60000 labels, 0-9)
//! - t10k-images-idx3-ubyte  (10000 test images)
//! - t10k-labels-idx1-ubyte  (10000 test labels)
//!
//! If files don't exist, generates synthetic MNIST-like data.
//!
//! ## Downloading real MNIST data
//!
//! To use real MNIST data, download the following files into your data directory:
//!
//! ```text
//! https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
//! https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
//! https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
//! https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz
//! ```
//!
//! Then decompress with `gunzip *.gz` and pass the directory to `MnistData::load()`.

use std::fmt;
use std::fs;
use std::path::Path;
use std::process::Command;

/// Errors that can occur during MNIST loading.
#[derive(Debug)]
pub enum MnistError {
    /// IDX file has invalid magic number.
    InvalidMagic { expected: u32, got: u32 },
    /// IO error reading file.
    Io(std::io::Error),
    /// Files not found and user must download manually.
    FilesNotFound {
        data_dir: String,
        missing: Vec<String>,
    },
    /// Download attempt failed.
    DownloadFailed(String),
}

impl fmt::Display for MnistError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MnistError::InvalidMagic { expected, got } => {
                write!(f, "Invalid IDX magic number: expected {expected}, got {got}")
            }
            MnistError::Io(e) => write!(f, "IO error: {e}"),
            MnistError::FilesNotFound { data_dir, missing } => {
                writeln!(f, "MNIST files not found in '{data_dir}'.")?;
                writeln!(f, "Missing files: {}", missing.join(", "))?;
                writeln!(f)?;
                writeln!(f, "Please download manually from:")?;
                for name in missing {
                    writeln!(
                        f,
                        "  https://storage.googleapis.com/cvdf-datasets/mnist/{name}.gz"
                    )?;
                }
                writeln!(f)?;
                write!(f, "Then decompress with: gunzip *.gz")
            }
            MnistError::DownloadFailed(msg) => {
                write!(f, "MNIST download failed: {msg}")
            }
        }
    }
}

impl std::error::Error for MnistError {}

impl From<std::io::Error> for MnistError {
    fn from(e: std::io::Error) -> Self {
        MnistError::Io(e)
    }
}

const MNIST_FILES: [&str; 4] = [
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
];

/// MNIST dataset.
pub struct MnistData {
    pub train_images: Vec<f32>, // [n_train, 784], normalized to [0, 1]
    pub train_labels: Vec<u8>,  // [n_train], values 0-9
    pub test_images: Vec<f32>,  // [n_test, 784]
    pub test_labels: Vec<u8>,   // [n_test]
    pub n_train: usize,
    pub n_test: usize,
    pub image_size: usize, // 784 = 28×28
    pub n_classes: usize,  // 10
}

/// Download instructions for MNIST data.
///
/// Because we avoid external HTTP dependencies, this function checks whether the
/// MNIST IDX files exist in `data_dir` and returns an error with download
/// instructions if any are missing.
pub fn download_mnist(data_dir: &str) -> Result<(), MnistError> {
    let dir = Path::new(data_dir);
    if !dir.exists() {
        fs::create_dir_all(dir)?;
    }

    let missing: Vec<String> = MNIST_FILES
        .iter()
        .filter(|name| !dir.join(name).exists())
        .map(|s| s.to_string())
        .collect();

    if missing.is_empty() {
        Ok(())
    } else {
        Err(MnistError::FilesNotFound {
            data_dir: data_dir.to_string(),
            missing,
        })
    }
}

const MNIST_BASE_URL: &str = "https://storage.googleapis.com/cvdf-datasets/mnist";

/// Download a single file using curl or wget (whichever is available).
///
/// Returns `Ok(())` on success. Handles HTTP redirects via curl `-L` or wget's
/// built-in redirect following.
fn download_file(url: &str, dest: &str) -> Result<(), MnistError> {
    // Try curl first (most common on Linux/macOS)
    let curl_result = Command::new("curl")
        .args(["-fSL", "--retry", "3", "-o", dest, url])
        .output();

    match curl_result {
        Ok(output) if output.status.success() => return Ok(()),
        _ => {}
    }

    // Fall back to wget
    let wget_result = Command::new("wget")
        .args(["-q", "-O", dest, url])
        .output();

    match wget_result {
        Ok(output) if output.status.success() => Ok(()),
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(MnistError::DownloadFailed(format!(
                "wget failed for {url}: {stderr}"
            )))
        }
        Err(_) => Err(MnistError::DownloadFailed(
            "Neither curl nor wget is available. Please install one of them \
             or download MNIST files manually."
                .to_string(),
        )),
    }
}

/// Decompress a .gz file in-place using gunzip or a gzip -d fallback.
///
/// After successful decompression the .gz file is removed (standard gunzip
/// behaviour).
fn decompress_gz(gz_path: &str) -> Result<(), MnistError> {
    // Try gunzip
    let result = Command::new("gunzip")
        .arg("-f")
        .arg(gz_path)
        .output();

    match result {
        Ok(output) if output.status.success() => return Ok(()),
        _ => {}
    }

    // Fall back to gzip -d
    let result = Command::new("gzip")
        .args(["-df", gz_path])
        .output();

    match result {
        Ok(output) if output.status.success() => Ok(()),
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(MnistError::DownloadFailed(format!(
                "Failed to decompress {gz_path}: {stderr}"
            )))
        }
        Err(_) => Err(MnistError::DownloadFailed(
            "Neither gunzip nor gzip is available. Please decompress .gz files manually."
                .to_string(),
        )),
    }
}

/// Download all 4 MNIST IDX files into `data_dir`, decompressing them.
///
/// Skips files that already exist (uncompressed). Uses `curl` or `wget` via
/// subprocess for HTTP, and `gunzip`/`gzip` for decompression.
pub fn download_mnist_files(data_dir: &str) -> Result<(), MnistError> {
    let dir = Path::new(data_dir);
    if !dir.exists() {
        fs::create_dir_all(dir)?;
    }

    for name in &MNIST_FILES {
        let dest = dir.join(name);
        if dest.exists() {
            continue;
        }

        let gz_name = format!("{name}.gz");
        let gz_dest = dir.join(&gz_name);
        let url = format!("{MNIST_BASE_URL}/{gz_name}");

        eprintln!("Downloading {url} ...");
        download_file(&url, gz_dest.to_str().unwrap_or(&gz_name))?;

        eprintln!("Decompressing {gz_name} ...");
        decompress_gz(gz_dest.to_str().unwrap_or(&gz_name))?;

        // Verify the decompressed file exists
        if !dest.exists() {
            return Err(MnistError::DownloadFailed(format!(
                "Decompressed file not found: {}",
                dest.display()
            )));
        }
    }

    Ok(())
}

impl MnistData {
    /// Download MNIST data (if not already present) and load it.
    ///
    /// This method:
    /// 1. Checks if all IDX files exist in `data_dir`
    /// 2. If not, downloads them from Google Storage using curl/wget
    /// 3. Decompresses the .gz files
    /// 4. Loads and returns the parsed dataset
    pub fn download_and_load(data_dir: &str) -> Result<Self, MnistError> {
        download_mnist_files(data_dir)?;
        Self::load_from_dir(data_dir)
    }

    /// Try to load real MNIST from IDX files in `data_dir`.
    ///
    /// Returns `Err` if any files are missing or malformed.
    /// Use this when you want explicit error handling.
    pub fn load_from_dir(data_dir: &str) -> Result<Self, MnistError> {
        // First check all files exist
        download_mnist(data_dir)?;

        let train_images_path = format!("{}/train-images-idx3-ubyte", data_dir);
        let train_labels_path = format!("{}/train-labels-idx1-ubyte", data_dir);
        let test_images_path = format!("{}/t10k-images-idx3-ubyte", data_dir);
        let test_labels_path = format!("{}/t10k-labels-idx1-ubyte", data_dir);

        let (train_images, n_train, rows, cols) = parse_idx_images(&train_images_path)?;
        let (train_labels, n_train_labels) = parse_idx_labels(&train_labels_path)?;
        let (test_images, n_test, _, _) = parse_idx_images(&test_images_path)?;
        let (test_labels, n_test_labels) = parse_idx_labels(&test_labels_path)?;

        let image_size = rows * cols;

        // Validate consistency
        if n_train != n_train_labels {
            return Err(MnistError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Train image count ({}) != label count ({})",
                    n_train, n_train_labels
                ),
            )));
        }
        if n_test != n_test_labels {
            return Err(MnistError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Test image count ({}) != label count ({})",
                    n_test, n_test_labels
                ),
            )));
        }

        Ok(Self {
            train_images,
            train_labels,
            test_images,
            test_labels,
            n_train,
            n_test,
            image_size,
            n_classes: 10,
        })
    }

    /// Load from IDX bytes directly (useful for testing or embedded data).
    ///
    /// Accepts raw IDX-format byte slices for each of the four MNIST files.
    pub fn from_idx_bytes(
        train_images_bytes: &[u8],
        train_labels_bytes: &[u8],
        test_images_bytes: &[u8],
        test_labels_bytes: &[u8],
    ) -> Result<Self, MnistError> {
        let (train_images, n_train, rows, cols) = parse_idx_images_bytes(train_images_bytes)?;
        let (train_labels, _) = parse_idx_labels_bytes(train_labels_bytes)?;
        let (test_images, n_test, _, _) = parse_idx_images_bytes(test_images_bytes)?;
        let (test_labels, _) = parse_idx_labels_bytes(test_labels_bytes)?;
        let image_size = rows * cols;

        Ok(Self {
            train_images,
            train_labels,
            test_images,
            test_labels,
            n_train,
            n_test,
            image_size,
            n_classes: 10,
        })
    }

    /// Try to load from IDX files, or fall back to synthetic data.
    ///
    /// This is the convenience method: it never fails, generating synthetic
    /// data if real MNIST files are not found.
    pub fn load(data_dir: &str) -> Self {
        match Self::load_from_dir(data_dir) {
            Ok(data) => data,
            Err(_) => Self::synthetic(1000, 200),
        }
    }

    /// Generate synthetic MNIST-like data with realistic variation.
    ///
    /// Each digit class has a distinct structural pattern. Variation is added
    /// through random translation offsets, stroke thickness jitter, and
    /// per-pixel noise so that no two samples are identical.
    pub fn synthetic(n_train: usize, n_test: usize) -> Self {
        let image_size = 784; // 28×28

        let mut train_images = vec![0.0f32; n_train * image_size];
        let mut train_labels = vec![0u8; n_train];
        let mut test_images = vec![0.0f32; n_test * image_size];
        let mut test_labels = vec![0u8; n_test];

        for i in 0..n_train {
            let label = (i % 10) as u8;
            train_labels[i] = label;
            draw_digit(
                &mut train_images[i * image_size..(i + 1) * image_size],
                label,
                i,
            );
        }

        for i in 0..n_test {
            let label = (i % 10) as u8;
            test_labels[i] = label;
            draw_digit(
                &mut test_images[i * image_size..(i + 1) * image_size],
                label,
                i + n_train,
            );
        }

        Self {
            train_images,
            train_labels,
            test_images,
            test_labels,
            n_train,
            n_test,
            image_size,
            n_classes: 10,
        }
    }

    /// Get a batch of training data.
    pub fn train_batch(&self, offset: usize, batch_size: usize) -> (&[f32], &[u8]) {
        let start = offset % self.n_train;
        let end = (start + batch_size).min(self.n_train);
        (
            &self.train_images[start * self.image_size..end * self.image_size],
            &self.train_labels[start..end],
        )
    }

    /// Return a human-readable summary of dataset statistics.
    pub fn summary(&self) -> String {
        let train_nonzero: usize = self.train_images.iter().filter(|&&x| x > 0.0).count();
        let test_nonzero: usize = self.test_images.iter().filter(|&&x| x > 0.0).count();
        let train_density = if self.n_train > 0 {
            train_nonzero as f64 / (self.n_train * self.image_size) as f64
        } else {
            0.0
        };
        let test_density = if self.n_test > 0 {
            test_nonzero as f64 / (self.n_test * self.image_size) as f64
        } else {
            0.0
        };

        let dist = self.class_distribution();
        let mut dist_str = String::new();
        for (label, count) in &dist {
            dist_str.push_str(&format!("  {}: {}\n", label, count));
        }

        format!(
            "MNIST Dataset Summary\n\
             =====================\n\
             Training samples: {}\n\
             Test samples:     {}\n\
             Image size:       {}x{} ({})\n\
             Classes:          {}\n\
             Train pixel density: {:.3}\n\
             Test pixel density:  {:.3}\n\
             \n\
             Class distribution (train):\n\
             {}",
            self.n_train,
            self.n_test,
            28,
            28,
            self.image_size,
            self.n_classes,
            train_density,
            test_density,
            dist_str,
        )
    }

    /// Render a sample image as ASCII art (28x28, using `#` for lit pixels and
    /// space for dark pixels).
    ///
    /// Uses the training set. Returns an empty string if `index` is out of range.
    pub fn visualize_sample(&self, index: usize) -> String {
        if index >= self.n_train {
            return String::new();
        }
        let start = index * self.image_size;
        let pixels = &self.train_images[start..start + self.image_size];
        let label = self.train_labels[index];

        let mut out = format!("Label: {}\n", label);
        for row in 0..28 {
            for col in 0..28 {
                let v = pixels[row * 28 + col];
                if v > 0.75 {
                    out.push('#');
                } else if v > 0.4 {
                    out.push('+');
                } else if v > 0.15 {
                    out.push('.');
                } else {
                    out.push(' ');
                }
            }
            out.push('\n');
        }
        out
    }

    /// Count the number of training samples per class label.
    ///
    /// Returns a vector of `(label, count)` pairs sorted by label 0..9.
    pub fn class_distribution(&self) -> Vec<(u8, usize)> {
        let mut counts = [0usize; 10];
        for &label in &self.train_labels {
            if (label as usize) < 10 {
                counts[label as usize] += 1;
            }
        }
        (0..10).map(|i| (i as u8, counts[i])).collect()
    }
}

// ---------------------------------------------------------------------------
// Simple deterministic PRNG (xorshift32) for reproducible noise
// ---------------------------------------------------------------------------

fn xorshift32(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

/// Draw a synthetic digit pattern on a 28×28 image with realistic variation.
///
/// Improvements over the basic version:
/// - Per-sample translation offset (random-ish via seed)
/// - Stroke thickness variation (draw neighbouring pixels with lower intensity)
/// - Per-pixel noise (random flips / intensity jitter)
/// - More structurally distinct patterns per class
fn draw_digit(image: &mut [f32], digit: u8, seed: usize) {
    let w = 28usize;

    // Deterministic pseudo-random state derived from seed
    let mut rng = (seed as u32).wrapping_mul(2654435761).wrapping_add(1);

    // Random offsets for translation variation (-2..+2)
    let offset_x = ((xorshift32(&mut rng) % 5) as i32) - 2;
    let offset_y = ((xorshift32(&mut rng) % 5) as i32) - 2;

    // Stroke thickness multiplier (0.7 .. 1.0)
    let thickness = 0.7 + (xorshift32(&mut rng) % 300) as f32 / 1000.0;

    // Safe pixel setter with anti-aliased thickness
    let set_thick = |img: &mut [f32], x: i32, y: i32, val: f32| {
        // Centre pixel
        if x >= 0 && x < w as i32 && y >= 0 && y < w as i32 {
            let idx = y as usize * w + x as usize;
            img[idx] = img[idx].max(val);
        }
        // Neighbouring pixels for thickness
        let neigh_val = val * thickness * 0.5;
        for &(dx, dy) in &[(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
            let nx = x + dx;
            let ny = y + dy;
            if nx >= 0 && nx < w as i32 && ny >= 0 && ny < w as i32 {
                let idx = ny as usize * w + nx as usize;
                img[idx] = img[idx].max(neigh_val);
            }
        }
    };

    let ox = offset_x;
    let oy = offset_y;

    match digit {
        0 => {
            // Ellipse (oval)
            for angle in 0..80 {
                let a = angle as f32 * std::f32::consts::PI * 2.0 / 80.0;
                let x = 14 + ox + (a.cos() * 7.0) as i32;
                let y = 14 + oy + (a.sin() * 9.0) as i32;
                set_thick(image, x, y, 1.0);
            }
        }
        1 => {
            // Vertical line with small serif at top and base
            for y in 5..23 {
                set_thick(image, 14 + ox, y + oy, 1.0);
            }
            // Top serif
            set_thick(image, 13 + ox, 6 + oy, 0.8);
            set_thick(image, 12 + ox, 7 + oy, 0.6);
            // Base
            for x in 12..17 {
                set_thick(image, x + ox, 23 + oy, 0.9);
            }
        }
        2 => {
            // Top arc + diagonal + bottom line
            for angle in 0..30 {
                let a = angle as f32 * std::f32::consts::PI / 30.0;
                let x = 14 + ox + (a.cos() * 6.0) as i32;
                let y = 9 + oy - (a.sin() * 4.0) as i32;
                set_thick(image, x, y, 1.0);
            }
            for i in 0..12 {
                let x = 20 + ox - i;
                let y = 9 + oy + i;
                set_thick(image, x, y, 1.0);
            }
            for x in 8..21 {
                set_thick(image, x + ox, 21 + oy, 1.0);
            }
        }
        3 => {
            // Three horizontal bars + right vertical
            for x in 10..20 {
                set_thick(image, x + ox, 6 + oy, 1.0);
                set_thick(image, x + ox, 13 + oy, 1.0);
                set_thick(image, x + ox, 21 + oy, 1.0);
            }
            for y in 6..22 {
                set_thick(image, 19 + ox, y + oy, 1.0);
            }
        }
        4 => {
            // Left vertical (top half) + horizontal bar + right vertical (full)
            for y in 4..14 {
                set_thick(image, 10 + ox, y + oy, 1.0);
            }
            for x in 10..21 {
                set_thick(image, x + ox, 14 + oy, 1.0);
            }
            for y in 4..24 {
                set_thick(image, 18 + ox, y + oy, 1.0);
            }
        }
        5 => {
            // Top bar (wide), left vertical (top half), middle bar, right vertical
            // (bottom half), bottom curved bar — shifted left relative to 3
            for x in 6..18 {
                set_thick(image, x + ox, 4 + oy, 1.0);
            }
            for y in 4..12 {
                set_thick(image, 6 + ox, y + oy, 1.0);
            }
            for x in 6..18 {
                set_thick(image, x + ox, 12 + oy, 1.0);
            }
            for y in 12..22 {
                set_thick(image, 17 + ox, y + oy, 1.0);
            }
            // Bottom curve
            for angle in 0..20 {
                let a = angle as f32 * std::f32::consts::PI / 20.0;
                let x = 12 + ox + (a.cos() * 5.0) as i32;
                let y = 22 + oy + (a.sin() * 2.0) as i32;
                set_thick(image, x, y, 1.0);
            }
        }
        6 => {
            // Left vertical full + bottom circle
            for y in 4..23 {
                set_thick(image, 10 + ox, y + oy, 1.0);
            }
            for angle in 0..50 {
                let a = angle as f32 * std::f32::consts::PI * 2.0 / 50.0;
                let x = 15 + ox + (a.cos() * 5.0) as i32;
                let y = 17 + oy + (a.sin() * 5.0) as i32;
                set_thick(image, x, y, 1.0);
            }
        }
        7 => {
            // Top bar + diagonal going down-left
            for x in 8..22 {
                set_thick(image, x + ox, 5 + oy, 1.0);
            }
            for i in 0..18 {
                let x = 21 + ox - (i * 2 / 3);
                let y = 5 + oy + i;
                set_thick(image, x, y, 1.0);
            }
        }
        8 => {
            // Two stacked circles
            for angle in 0..50 {
                let a = angle as f32 * std::f32::consts::PI * 2.0 / 50.0;
                let x_top = 14 + ox + (a.cos() * 5.0) as i32;
                let y_top = 9 + oy + (a.sin() * 4.0) as i32;
                set_thick(image, x_top, y_top, 1.0);
                let x_bot = 14 + ox + (a.cos() * 5.0) as i32;
                let y_bot = 19 + oy + (a.sin() * 4.0) as i32;
                set_thick(image, x_bot, y_bot, 1.0);
            }
        }
        9 => {
            // Top circle + right vertical
            for angle in 0..50 {
                let a = angle as f32 * std::f32::consts::PI * 2.0 / 50.0;
                let x = 14 + ox + (a.cos() * 5.0) as i32;
                let y = 10 + oy + (a.sin() * 5.0) as i32;
                set_thick(image, x, y, 1.0);
            }
            for y in 10..24 {
                set_thick(image, 19 + ox, y + oy, 1.0);
            }
        }
        _ => {}
    }

    // Add per-pixel noise: randomly flip some pixels and add intensity jitter
    for i in 0..image.len() {
        let r = xorshift32(&mut rng);
        // ~3% chance to flip a dark pixel to faint
        if image[i] < 0.1 && r % 33 == 0 {
            image[i] = 0.15 + (r % 100) as f32 / 500.0;
        }
        // Add slight intensity jitter to lit pixels
        if image[i] > 0.3 {
            let jitter = ((r >> 8) % 100) as f32 / 500.0 - 0.1; // -0.1 .. +0.1
            image[i] = (image[i] + jitter).clamp(0.0, 1.0);
        }
    }
}

/// Read a big-endian u32 from a byte slice at the given offset.
fn read_u32_be(data: &[u8], offset: usize) -> Result<u32, MnistError> {
    if data.len() < offset + 4 {
        return Err(MnistError::Io(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            format!("IDX file too short: need {} bytes, have {}", offset + 4, data.len()),
        )));
    }
    Ok(u32::from_be_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]))
}

/// Parse IDX image file format (magic 2051).
///
/// Format: [magic:u32] [n_images:u32] [n_rows:u32] [n_cols:u32] [pixel_data:u8...]
/// Returns `(images_f32, n_images, n_rows, n_cols)`.
fn parse_idx_images(path: &str) -> Result<(Vec<f32>, usize, usize, usize), MnistError> {
    let data = fs::read(path)?;
    parse_idx_images_bytes(&data)
}

/// Parse IDX image data from raw bytes.
fn parse_idx_images_bytes(data: &[u8]) -> Result<(Vec<f32>, usize, usize, usize), MnistError> {
    let magic = read_u32_be(data, 0)?;
    if magic != 2051 {
        return Err(MnistError::InvalidMagic {
            expected: 2051,
            got: magic,
        });
    }

    let n_images = read_u32_be(data, 4)? as usize;
    let n_rows = read_u32_be(data, 8)? as usize;
    let n_cols = read_u32_be(data, 12)? as usize;
    let pixel_count = n_images * n_rows * n_cols;
    let header_size = 16;

    if data.len() < header_size + pixel_count {
        return Err(MnistError::Io(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            format!(
                "IDX image file truncated: expected {} pixel bytes, have {}",
                pixel_count,
                data.len() - header_size
            ),
        )));
    }

    let pixels = &data[header_size..header_size + pixel_count];
    let images: Vec<f32> = pixels.iter().map(|&b| b as f32 / 255.0).collect();
    Ok((images, n_images, n_rows, n_cols))
}

/// Parse IDX label file format (magic 2049).
///
/// Format: [magic:u32] [n_labels:u32] [label_data:u8...]
fn parse_idx_labels(path: &str) -> Result<(Vec<u8>, usize), MnistError> {
    let data = fs::read(path)?;
    parse_idx_labels_bytes(&data)
}

/// Parse IDX label data from raw bytes.
fn parse_idx_labels_bytes(data: &[u8]) -> Result<(Vec<u8>, usize), MnistError> {
    let magic = read_u32_be(data, 0)?;
    if magic != 2049 {
        return Err(MnistError::InvalidMagic {
            expected: 2049,
            got: magic,
        });
    }

    let n_labels = read_u32_be(data, 4)? as usize;
    let header_size = 8;

    if data.len() < header_size + n_labels {
        return Err(MnistError::Io(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            format!(
                "IDX label file truncated: expected {} label bytes, have {}",
                n_labels,
                data.len() - header_size
            ),
        )));
    }

    let labels = data[header_size..header_size + n_labels].to_vec();
    Ok((labels, n_labels))
}

/// Create an IDX image file in memory (for testing).
///
/// Returns the raw bytes of a valid IDX3 file.
pub fn create_idx_images(images: &[u8], n_images: u32, n_rows: u32, n_cols: u32) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&2051u32.to_be_bytes());
    data.extend_from_slice(&n_images.to_be_bytes());
    data.extend_from_slice(&n_rows.to_be_bytes());
    data.extend_from_slice(&n_cols.to_be_bytes());
    data.extend_from_slice(images);
    data
}

/// Create an IDX label file in memory (for testing).
///
/// Returns the raw bytes of a valid IDX1 file.
pub fn create_idx_labels(labels: &[u8], n_labels: u32) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&2049u32.to_be_bytes());
    data.extend_from_slice(&n_labels.to_be_bytes());
    data.extend_from_slice(labels);
    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_data_has_correct_sizes() {
        let data = MnistData::synthetic(100, 20);
        assert_eq!(data.n_train, 100);
        assert_eq!(data.n_test, 20);
        assert_eq!(data.train_images.len(), 100 * 784);
        assert_eq!(data.train_labels.len(), 100);
        assert_eq!(data.test_images.len(), 20 * 784);
        assert_eq!(data.test_labels.len(), 20);
        assert_eq!(data.image_size, 784);
        assert_eq!(data.n_classes, 10);
        assert!(data.train_labels.iter().all(|&l| l < 10));
        assert!(data.test_labels.iter().all(|&l| l < 10));
    }

    #[test]
    fn all_ten_classes_present() {
        let data = MnistData::synthetic(100, 20);
        let dist = data.class_distribution();
        assert_eq!(dist.len(), 10);
        for (label, count) in &dist {
            assert!(*label < 10, "unexpected label {}", label);
            assert!(*count > 0, "class {} has zero samples", label);
        }
    }

    #[test]
    fn patterns_distinct_between_classes() {
        // Each digit should produce a different pattern
        let mut patterns: Vec<Vec<u8>> = Vec::new();
        for d in 0..10u8 {
            let mut img = vec![0.0f32; 784];
            draw_digit(&mut img, d, 42); // same seed so only digit differs
            let binary: Vec<u8> = img.iter().map(|&x| if x > 0.5 { 1 } else { 0 }).collect();
            patterns.push(binary);
        }

        for i in 0..10 {
            for j in (i + 1)..10 {
                assert_ne!(
                    patterns[i], patterns[j],
                    "Digit {i} and {j} have identical patterns"
                );
            }
        }

        // Additionally verify that patterns differ substantially (>5% of pixels)
        for i in 0..10 {
            for j in (i + 1)..10 {
                let diff: usize = patterns[i]
                    .iter()
                    .zip(patterns[j].iter())
                    .filter(|(a, b)| a != b)
                    .count();
                assert!(
                    diff > 784 / 20,
                    "Digits {} and {} differ by only {} pixels",
                    i,
                    j,
                    diff
                );
            }
        }
    }

    #[test]
    fn visualization_produces_nonempty_string() {
        let data = MnistData::synthetic(20, 5);
        let viz = data.visualize_sample(0);
        assert!(!viz.is_empty(), "visualization should be non-empty");
        assert!(viz.contains("Label:"), "should contain label header");
        // Should have 28 rows of content (plus label line)
        let lines: Vec<&str> = viz.lines().collect();
        assert_eq!(lines.len(), 29, "expected 1 label line + 28 image rows");
        // At least some lit pixels
        assert!(
            viz.contains('#') || viz.contains('+'),
            "visualization should contain lit pixels"
        );

        // Out of range returns empty
        let empty = data.visualize_sample(9999);
        assert!(empty.is_empty());
    }

    #[test]
    fn class_distribution_is_balanced() {
        let data = MnistData::synthetic(1000, 200);
        let dist = data.class_distribution();
        assert_eq!(dist.len(), 10);

        // With n_train=1000 and label = i%10, each class should have exactly 100
        for (label, count) in &dist {
            assert_eq!(
                *count, 100,
                "class {} should have 100 samples but has {}",
                label, count
            );
        }
    }

    #[test]
    fn batch_loading() {
        let data = MnistData::synthetic(100, 20);
        let (images, labels) = data.train_batch(0, 10);
        assert_eq!(images.len(), 10 * 784);
        assert_eq!(labels.len(), 10);
    }

    #[test]
    fn summary_contains_key_info() {
        let data = MnistData::synthetic(100, 20);
        let s = data.summary();
        assert!(s.contains("100"), "summary should mention train count");
        assert!(s.contains("20"), "summary should mention test count");
        assert!(s.contains("28"), "summary should mention image dimensions");
        assert!(
            s.contains("Class distribution"),
            "summary should show distribution"
        );
    }

    #[test]
    fn download_mnist_reports_missing_files() {
        // Use a temp dir that won't have MNIST files
        let dir = "/tmp/qlang_test_mnist_missing";
        let _ = fs::remove_dir_all(dir);
        let result = download_mnist(dir);
        assert!(result.is_err());
        match result.unwrap_err() {
            MnistError::FilesNotFound { missing, .. } => {
                assert_eq!(missing.len(), 4);
            }
            other => panic!("Expected FilesNotFound, got: {other}"),
        }
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn synthetic_images_have_variation() {
        // Two samples of the same digit with different seeds should differ
        let mut img1 = vec![0.0f32; 784];
        let mut img2 = vec![0.0f32; 784];
        draw_digit(&mut img1, 3, 0);
        draw_digit(&mut img2, 3, 50);

        let diff: usize = img1
            .iter()
            .zip(img2.iter())
            .filter(|(a, b)| ((*a) - (*b)).abs() > 0.05)
            .count();
        assert!(
            diff > 10,
            "same digit with different seeds should vary, but diff={}",
            diff
        );
    }

    // -----------------------------------------------------------------------
    // IDX format parsing tests
    // -----------------------------------------------------------------------

    #[test]
    fn parse_idx_images_bytes_valid() {
        // Create a tiny 2-image, 2x3 pixel IDX file
        let pixels: Vec<u8> = vec![0, 128, 255, 64, 192, 32, 10, 20, 30, 40, 50, 60];
        let data = create_idx_images(&pixels, 2, 2, 3);
        let (images, n, rows, cols) = parse_idx_images_bytes(&data).unwrap();

        assert_eq!(n, 2);
        assert_eq!(rows, 2);
        assert_eq!(cols, 3);
        assert_eq!(images.len(), 12);
        // Check normalization
        assert!((images[0] - 0.0).abs() < 1e-6); // pixel 0
        assert!((images[1] - 128.0 / 255.0).abs() < 1e-4); // pixel 128
        assert!((images[2] - 1.0).abs() < 1e-6); // pixel 255
    }

    #[test]
    fn parse_idx_labels_bytes_valid() {
        let labels: Vec<u8> = vec![0, 3, 7, 9, 1];
        let data = create_idx_labels(&labels, 5);
        let (parsed, n) = parse_idx_labels_bytes(&data).unwrap();

        assert_eq!(n, 5);
        assert_eq!(parsed, vec![0, 3, 7, 9, 1]);
    }

    #[test]
    fn parse_idx_images_wrong_magic() {
        let mut data = create_idx_images(&[0; 4], 1, 2, 2);
        // Corrupt the magic number
        data[3] = 0xFF;
        let result = parse_idx_images_bytes(&data);
        assert!(result.is_err());
        match result.unwrap_err() {
            MnistError::InvalidMagic { expected, .. } => assert_eq!(expected, 2051),
            other => panic!("Expected InvalidMagic, got: {other}"),
        }
    }

    #[test]
    fn parse_idx_labels_wrong_magic() {
        let mut data = create_idx_labels(&[0; 3], 3);
        data[3] = 0xFF;
        let result = parse_idx_labels_bytes(&data);
        assert!(result.is_err());
        match result.unwrap_err() {
            MnistError::InvalidMagic { expected, .. } => assert_eq!(expected, 2049),
            other => panic!("Expected InvalidMagic, got: {other}"),
        }
    }

    #[test]
    fn parse_idx_images_truncated() {
        // Claim 10 images but only provide 1 image worth of pixels
        let pixels = vec![0u8; 4]; // 1 image of 2x2
        let data = create_idx_images(&pixels, 10, 2, 2);
        let result = parse_idx_images_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    fn parse_idx_labels_truncated() {
        let labels = vec![0u8; 2];
        let data = create_idx_labels(&labels, 100); // claim 100 but only 2
        let result = parse_idx_labels_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    fn parse_idx_too_short_for_header() {
        // Only 3 bytes -- can't even read magic
        let result = parse_idx_images_bytes(&[0, 0, 0]);
        assert!(result.is_err());

        let result = parse_idx_labels_bytes(&[0, 0, 0]);
        assert!(result.is_err());
    }

    #[test]
    fn from_idx_bytes_roundtrip() {
        // Create small MNIST-like dataset: 5 train, 3 test, 4x4 images
        let n_train = 5u32;
        let n_test = 3u32;
        let rows = 4u32;
        let cols = 4u32;
        let img_size = (rows * cols) as usize;

        let train_pixels: Vec<u8> = (0..n_train as usize * img_size)
            .map(|i| (i % 256) as u8)
            .collect();
        let train_labels: Vec<u8> = (0..n_train as usize).map(|i| (i % 10) as u8).collect();
        let test_pixels: Vec<u8> = (0..n_test as usize * img_size)
            .map(|i| ((i + 100) % 256) as u8)
            .collect();
        let test_labels: Vec<u8> = (0..n_test as usize).map(|i| ((i + 5) % 10) as u8).collect();

        let train_img_bytes = create_idx_images(&train_pixels, n_train, rows, cols);
        let train_lbl_bytes = create_idx_labels(&train_labels, n_train);
        let test_img_bytes = create_idx_images(&test_pixels, n_test, rows, cols);
        let test_lbl_bytes = create_idx_labels(&test_labels, n_test);

        let data = MnistData::from_idx_bytes(
            &train_img_bytes,
            &train_lbl_bytes,
            &test_img_bytes,
            &test_lbl_bytes,
        )
        .unwrap();

        assert_eq!(data.n_train, 5);
        assert_eq!(data.n_test, 3);
        assert_eq!(data.image_size, 16); // 4x4
        assert_eq!(data.train_images.len(), 5 * 16);
        assert_eq!(data.test_images.len(), 3 * 16);
        assert_eq!(data.train_labels, train_labels);
        assert_eq!(data.test_labels, test_labels);

        // Check pixel normalization
        assert!((data.train_images[0] - 0.0 / 255.0).abs() < 1e-6);
        assert!((data.train_images[1] - 1.0 / 255.0).abs() < 1e-4);
    }

    #[test]
    fn load_from_dir_with_real_idx_files() {
        // Write proper IDX files to a temp directory and load them
        let dir = "/tmp/qlang_test_mnist_idx_load";
        let _ = fs::remove_dir_all(dir);
        fs::create_dir_all(dir).unwrap();

        let n_train = 10u32;
        let n_test = 5u32;
        let rows = 28u32;
        let cols = 28u32;
        let img_size = (rows * cols) as usize;

        // Generate pixels with some structure
        let train_pixels: Vec<u8> = (0..n_train as usize * img_size)
            .map(|i| (i % 256) as u8)
            .collect();
        let train_labels: Vec<u8> = (0..n_train as usize).map(|i| (i % 10) as u8).collect();
        let test_pixels: Vec<u8> = (0..n_test as usize * img_size)
            .map(|i| ((i * 7) % 256) as u8)
            .collect();
        let test_labels: Vec<u8> = (0..n_test as usize).map(|i| (i % 10) as u8).collect();

        // Write IDX files
        fs::write(
            format!("{dir}/train-images-idx3-ubyte"),
            create_idx_images(&train_pixels, n_train, rows, cols),
        )
        .unwrap();
        fs::write(
            format!("{dir}/train-labels-idx1-ubyte"),
            create_idx_labels(&train_labels, n_train),
        )
        .unwrap();
        fs::write(
            format!("{dir}/t10k-images-idx3-ubyte"),
            create_idx_images(&test_pixels, n_test, rows, cols),
        )
        .unwrap();
        fs::write(
            format!("{dir}/t10k-labels-idx1-ubyte"),
            create_idx_labels(&test_labels, n_test),
        )
        .unwrap();

        // Load and verify
        let data = MnistData::load_from_dir(dir).unwrap();
        assert_eq!(data.n_train, 10);
        assert_eq!(data.n_test, 5);
        assert_eq!(data.image_size, 784);
        assert_eq!(data.train_labels.len(), 10);
        assert_eq!(data.test_labels.len(), 5);
        assert_eq!(data.train_images.len(), 10 * 784);
        assert_eq!(data.test_images.len(), 5 * 784);

        // Verify labels roundtripped correctly
        for i in 0..10 {
            assert_eq!(data.train_labels[i], (i % 10) as u8);
        }

        // Verify pixel normalization
        assert!((data.train_images[0] - 0.0).abs() < 1e-6);
        assert!((data.train_images[1] - 1.0 / 255.0).abs() < 1e-4);

        // Clean up
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn load_from_dir_missing_files_returns_error() {
        let dir = "/tmp/qlang_test_mnist_idx_missing";
        let _ = fs::remove_dir_all(dir);
        let result = MnistData::load_from_dir(dir);
        assert!(result.is_err());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn load_falls_back_to_synthetic() {
        // load() should return synthetic data when directory doesn't exist
        let data = MnistData::load("/tmp/qlang_nonexistent_dir_12345");
        assert_eq!(data.n_train, 1000);
        assert_eq!(data.n_test, 200);
        assert_eq!(data.image_size, 784);
    }

    #[test]
    fn load_uses_real_files_when_present() {
        let dir = "/tmp/qlang_test_mnist_load_real";
        let _ = fs::remove_dir_all(dir);
        fs::create_dir_all(dir).unwrap();

        let n_train = 20u32;
        let n_test = 8u32;

        let train_pixels = vec![128u8; n_train as usize * 784];
        let train_labels: Vec<u8> = (0..n_train as usize).map(|i| (i % 10) as u8).collect();
        let test_pixels = vec![64u8; n_test as usize * 784];
        let test_labels: Vec<u8> = (0..n_test as usize).map(|i| (i % 10) as u8).collect();

        fs::write(
            format!("{dir}/train-images-idx3-ubyte"),
            create_idx_images(&train_pixels, n_train, 28, 28),
        )
        .unwrap();
        fs::write(
            format!("{dir}/train-labels-idx1-ubyte"),
            create_idx_labels(&train_labels, n_train),
        )
        .unwrap();
        fs::write(
            format!("{dir}/t10k-images-idx3-ubyte"),
            create_idx_images(&test_pixels, n_test, 28, 28),
        )
        .unwrap();
        fs::write(
            format!("{dir}/t10k-labels-idx1-ubyte"),
            create_idx_labels(&test_labels, n_test),
        )
        .unwrap();

        let data = MnistData::load(dir);
        // Should use real data, not synthetic defaults (1000/200)
        assert_eq!(data.n_train, 20);
        assert_eq!(data.n_test, 8);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn create_idx_images_format_correctness() {
        let pixels = vec![42u8; 6]; // 2 images of 1x3
        let data = create_idx_images(&pixels, 2, 1, 3);

        // Verify header
        assert_eq!(read_u32_be(&data, 0).unwrap(), 2051); // magic
        assert_eq!(read_u32_be(&data, 4).unwrap(), 2); // n_images
        assert_eq!(read_u32_be(&data, 8).unwrap(), 1); // rows
        assert_eq!(read_u32_be(&data, 12).unwrap(), 3); // cols
        assert_eq!(data.len(), 16 + 6); // header + pixels
        assert_eq!(data[16], 42); // first pixel
    }

    #[test]
    fn create_idx_labels_format_correctness() {
        let labels = vec![5, 3, 7];
        let data = create_idx_labels(&labels, 3);

        assert_eq!(read_u32_be(&data, 0).unwrap(), 2049); // magic
        assert_eq!(read_u32_be(&data, 4).unwrap(), 3); // n_labels
        assert_eq!(data.len(), 8 + 3); // header + labels
        assert_eq!(data[8], 5);
        assert_eq!(data[9], 3);
        assert_eq!(data[10], 7);
    }

    #[test]
    fn mnist_error_display() {
        let err = MnistError::InvalidMagic {
            expected: 2051,
            got: 9999,
        };
        let msg = format!("{err}");
        assert!(msg.contains("2051"));
        assert!(msg.contains("9999"));

        let err = MnistError::FilesNotFound {
            data_dir: "/some/dir".into(),
            missing: vec!["train-images-idx3-ubyte".into()],
        };
        let msg = format!("{err}");
        assert!(msg.contains("/some/dir"));
        assert!(msg.contains("train-images-idx3-ubyte"));
        assert!(msg.contains("download"));

        let err = MnistError::DownloadFailed("no curl".into());
        let msg = format!("{err}");
        assert!(msg.contains("no curl"));
    }

    #[test]
    fn download_and_load_with_preexisting_files() {
        // Simulate download_and_load by pre-creating IDX files in a temp dir.
        // This tests the full flow without needing real network access.
        let dir = "/tmp/qlang_test_download_and_load";
        let _ = fs::remove_dir_all(dir);
        fs::create_dir_all(dir).unwrap();

        let n_train = 30u32;
        let n_test = 10u32;
        let rows = 28u32;
        let cols = 28u32;
        let img_size = (rows * cols) as usize;

        let train_pixels: Vec<u8> = (0..n_train as usize * img_size)
            .map(|i| ((i * 3) % 256) as u8)
            .collect();
        let train_labels: Vec<u8> = (0..n_train as usize).map(|i| (i % 10) as u8).collect();
        let test_pixels: Vec<u8> = (0..n_test as usize * img_size)
            .map(|i| ((i * 7 + 50) % 256) as u8)
            .collect();
        let test_labels: Vec<u8> = (0..n_test as usize).map(|i| (i % 10) as u8).collect();

        fs::write(
            format!("{dir}/train-images-idx3-ubyte"),
            create_idx_images(&train_pixels, n_train, rows, cols),
        )
        .unwrap();
        fs::write(
            format!("{dir}/train-labels-idx1-ubyte"),
            create_idx_labels(&train_labels, n_train),
        )
        .unwrap();
        fs::write(
            format!("{dir}/t10k-images-idx3-ubyte"),
            create_idx_images(&test_pixels, n_test, rows, cols),
        )
        .unwrap();
        fs::write(
            format!("{dir}/t10k-labels-idx1-ubyte"),
            create_idx_labels(&test_labels, n_test),
        )
        .unwrap();

        // download_and_load should succeed because files already exist
        let data = MnistData::download_and_load(dir).unwrap();
        assert_eq!(data.n_train, 30);
        assert_eq!(data.n_test, 10);
        assert_eq!(data.image_size, 784);

        let _ = fs::remove_dir_all(dir);
    }

    /// End-to-end test: create synthetic IDX data, save to disk, load it back,
    /// train a small MLP with `train_step_backprop()`, and verify accuracy > 90%.
    #[test]
    fn idx_roundtrip_train_and_verify_accuracy() {
        use crate::training::MlpWeights;

        let dir = "/tmp/qlang_test_idx_roundtrip_train";
        let _ = fs::remove_dir_all(dir);
        fs::create_dir_all(dir).unwrap();

        // Generate synthetic images with distinct per-class patterns.
        // We use 4 classes (0-3) and 8x8 images (64 pixels) to keep it fast.
        let rows = 8u32;
        let cols = 8u32;
        let img_size = (rows * cols) as usize;
        let n_classes = 4u8;
        let n_train = 200u32;
        let n_test = 40u32;

        let mut rng_state: u32 = 12345;
        let mut next_rand = || -> u32 {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 17;
            rng_state ^= rng_state << 5;
            rng_state
        };

        // Create training data with clear patterns per class
        let mut train_pixels = vec![0u8; n_train as usize * img_size];
        let mut train_labels = vec![0u8; n_train as usize];

        for i in 0..n_train as usize {
            let label = (i % n_classes as usize) as u8;
            train_labels[i] = label;
            let offset = i * img_size;

            match label {
                0 => {
                    // Horizontal bar in top half
                    for c in 0..cols as usize {
                        train_pixels[offset + 2 * cols as usize + c] = 200;
                        train_pixels[offset + 3 * cols as usize + c] = 200;
                    }
                }
                1 => {
                    // Vertical bar on left side
                    for r in 0..rows as usize {
                        train_pixels[offset + r * cols as usize + 2] = 200;
                        train_pixels[offset + r * cols as usize + 3] = 200;
                    }
                }
                2 => {
                    // Diagonal top-left to bottom-right
                    for k in 0..rows.min(cols) as usize {
                        train_pixels[offset + k * cols as usize + k] = 200;
                    }
                }
                3 => {
                    // Bottom-right quadrant filled
                    for r in 4..rows as usize {
                        for c in 4..cols as usize {
                            train_pixels[offset + r * cols as usize + c] = 200;
                        }
                    }
                }
                _ => {}
            }

            // Small noise
            for j in 0..img_size {
                let noise = (next_rand() % 20) as u8;
                train_pixels[offset + j] = train_pixels[offset + j].saturating_add(noise);
            }
        }

        // Create test data with the same patterns
        let mut test_pixels = vec![0u8; n_test as usize * img_size];
        let mut test_labels = vec![0u8; n_test as usize];

        for i in 0..n_test as usize {
            let label = (i % n_classes as usize) as u8;
            test_labels[i] = label;
            let offset = i * img_size;

            match label {
                0 => {
                    for c in 0..cols as usize {
                        test_pixels[offset + 2 * cols as usize + c] = 200;
                        test_pixels[offset + 3 * cols as usize + c] = 200;
                    }
                }
                1 => {
                    for r in 0..rows as usize {
                        test_pixels[offset + r * cols as usize + 2] = 200;
                        test_pixels[offset + r * cols as usize + 3] = 200;
                    }
                }
                2 => {
                    for k in 0..rows.min(cols) as usize {
                        test_pixels[offset + k * cols as usize + k] = 200;
                    }
                }
                3 => {
                    for r in 4..rows as usize {
                        for c in 4..cols as usize {
                            test_pixels[offset + r * cols as usize + c] = 200;
                        }
                    }
                }
                _ => {}
            }

            for j in 0..img_size {
                let noise = (next_rand() % 20) as u8;
                test_pixels[offset + j] = test_pixels[offset + j].saturating_add(noise);
            }
        }

        // Write IDX files to disk
        fs::write(
            format!("{dir}/train-images-idx3-ubyte"),
            create_idx_images(&train_pixels, n_train, rows, cols),
        )
        .unwrap();
        fs::write(
            format!("{dir}/train-labels-idx1-ubyte"),
            create_idx_labels(&train_labels, n_train),
        )
        .unwrap();
        fs::write(
            format!("{dir}/t10k-images-idx3-ubyte"),
            create_idx_images(&test_pixels, n_test, rows, cols),
        )
        .unwrap();
        fs::write(
            format!("{dir}/t10k-labels-idx1-ubyte"),
            create_idx_labels(&test_labels, n_test),
        )
        .unwrap();

        // Load back from IDX files
        let data = MnistData::load_from_dir(dir).unwrap();
        assert_eq!(data.n_train, n_train as usize);
        assert_eq!(data.n_test, n_test as usize);
        assert_eq!(data.image_size, img_size);

        // Train a small MLP using backprop
        let mut mlp = MlpWeights::new(img_size, 32, n_classes as usize);

        let batch_size = 40;
        for epoch in 0..80 {
            for batch_start in (0..data.n_train).step_by(batch_size) {
                let (batch_images, batch_labels) = data.train_batch(batch_start, batch_size);
                mlp.train_step_backprop(batch_images, batch_labels, 0.05);
            }

            if epoch % 20 == 0 {
                let probs = mlp.forward(&data.train_images);
                let acc = mlp.accuracy(&probs, &data.train_labels);
                eprintln!("  epoch {epoch}: train_acc = {:.1}%", acc * 100.0);
            }
        }

        // Evaluate on training set
        let train_probs = mlp.forward(&data.train_images);
        let train_acc = mlp.accuracy(&train_probs, &data.train_labels);

        // Evaluate on test set
        let test_probs = mlp.forward(&data.test_images);
        let test_acc = mlp.accuracy(&test_probs, &data.test_labels);

        eprintln!(
            "  Final: train_acc = {:.1}%, test_acc = {:.1}%",
            train_acc * 100.0,
            test_acc * 100.0,
        );

        assert!(
            train_acc > 0.90,
            "Expected >90% train accuracy, got {:.1}%",
            train_acc * 100.0
        );
        assert!(
            test_acc > 0.90,
            "Expected >90% test accuracy, got {:.1}%",
            test_acc * 100.0
        );

        // Clean up
        let _ = fs::remove_dir_all(dir);
    }
}
