use serde::{Deserialize, Serialize};
use std::fmt;

/// Data type of tensor elements — what the CPU registers actually hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Dtype {
    F16,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    Bool,
    /// Ternary: {-1, 0, +1} — core to IGQK compression
    Ternary,
    /// UTF-8 encoded text
    Utf8,
}

impl Dtype {
    /// Size in bytes of a single element.
    pub fn size_bytes(self) -> usize {
        match self {
            Dtype::Bool => 1,
            Dtype::I8 => 1,
            Dtype::Ternary => 1, // stored as i8, uses only {-1, 0, 1}
            Dtype::Utf8 => 1,    // per-byte; length determined by shape
            Dtype::F16 | Dtype::I16 => 2,
            Dtype::F32 | Dtype::I32 => 4,
            Dtype::F64 | Dtype::I64 => 8,
        }
    }
}

impl fmt::Display for Dtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dtype::F16 => write!(f, "f16"),
            Dtype::F32 => write!(f, "f32"),
            Dtype::F64 => write!(f, "f64"),
            Dtype::I8 => write!(f, "i8"),
            Dtype::I16 => write!(f, "i16"),
            Dtype::I32 => write!(f, "i32"),
            Dtype::I64 => write!(f, "i64"),
            Dtype::Bool => write!(f, "bool"),
            Dtype::Ternary => write!(f, "ternary"),
            Dtype::Utf8 => write!(f, "utf8"),
        }
    }
}

/// Shape of a tensor: list of dimension sizes.
/// Empty = scalar. [n] = vector. [m, n] = matrix.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Shape(pub Vec<Dim>);

/// A single dimension — either a known size or dynamic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Dim {
    Fixed(usize),
    Dynamic, // unknown at compile time, resolved at runtime
}

impl Shape {
    pub fn scalar() -> Self {
        Shape(vec![])
    }

    pub fn vector(n: usize) -> Self {
        Shape(vec![Dim::Fixed(n)])
    }

    pub fn matrix(m: usize, n: usize) -> Self {
        Shape(vec![Dim::Fixed(m), Dim::Fixed(n)])
    }

    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// Total number of elements, if all dimensions are known.
    pub fn numel(&self) -> Option<usize> {
        let mut total = 1usize;
        for d in &self.0 {
            match d {
                Dim::Fixed(n) => total = total.checked_mul(*n)?,
                Dim::Dynamic => return None,
            }
        }
        Some(total)
    }

    /// Check if this shape is compatible with another (for broadcasting or connection).
    pub fn is_compatible_with(&self, other: &Shape) -> bool {
        if self.rank() != other.rank() {
            return false;
        }
        self.0.iter().zip(other.0.iter()).all(|(a, b)| match (a, b) {
            (Dim::Fixed(x), Dim::Fixed(y)) => x == y,
            _ => true, // dynamic dims are compatible with anything
        })
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            match d {
                Dim::Fixed(n) => write!(f, "{n}")?,
                Dim::Dynamic => write!(f, "?")?,
            }
        }
        write!(f, "]")
    }
}

/// Complete tensor type: dtype + shape.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorType {
    pub dtype: Dtype,
    pub shape: Shape,
}

impl TensorType {
    pub fn new(dtype: Dtype, shape: Shape) -> Self {
        Self { dtype, shape }
    }

    pub fn f32_scalar() -> Self {
        Self::new(Dtype::F32, Shape::scalar())
    }

    pub fn f32_vector(n: usize) -> Self {
        Self::new(Dtype::F32, Shape::vector(n))
    }

    pub fn f32_matrix(m: usize, n: usize) -> Self {
        Self::new(Dtype::F32, Shape::matrix(m, n))
    }

    pub fn ternary_matrix(m: usize, n: usize) -> Self {
        Self::new(Dtype::Ternary, Shape::matrix(m, n))
    }

    /// Total memory in bytes.
    pub fn size_bytes(&self) -> Option<usize> {
        self.shape.numel().map(|n| n * self.dtype.size_bytes())
    }
}

impl fmt::Display for TensorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor<{}>{}", self.dtype, self.shape)
    }
}

/// Concrete tensor data — the actual values in memory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorData {
    pub dtype: Dtype,
    pub shape: Shape,
    pub data: Vec<u8>, // raw bytes, interpreted according to dtype
}

impl TensorData {
    /// Create a tensor filled with zeros.
    pub fn zeros(tensor_type: &TensorType) -> Option<Self> {
        let n_bytes = tensor_type.size_bytes()?;
        Some(Self {
            dtype: tensor_type.dtype,
            shape: tensor_type.shape.clone(),
            data: vec![0u8; n_bytes],
        })
    }

    /// Create a f32 tensor from a slice.
    pub fn from_f32(shape: Shape, values: &[f32]) -> Self {
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self {
            dtype: Dtype::F32,
            shape,
            data,
        }
    }

    /// Read f32 values from this tensor.
    pub fn as_f32_slice(&self) -> Option<Vec<f32>> {
        if self.dtype != Dtype::F32 {
            return None;
        }
        Some(
            self.data
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect(),
        )
    }

    /// Create a UTF-8 text tensor from a string.
    pub fn from_string(s: &str) -> Self {
        TensorData {
            dtype: Dtype::Utf8,
            shape: Shape(vec![Dim::Fixed(s.len())]),
            data: s.as_bytes().to_vec(),
        }
    }

    /// Read this tensor as a UTF-8 string (returns `None` if dtype is not Utf8
    /// or the bytes are not valid UTF-8).
    pub fn as_string(&self) -> Option<String> {
        if self.dtype == Dtype::Utf8 {
            String::from_utf8(self.data.clone()).ok()
        } else {
            None
        }
    }

    /// Get the tensor type.
    pub fn tensor_type(&self) -> TensorType {
        TensorType::new(self.dtype, self.shape.clone())
    }

    // ------------------------------------------------------------------
    // Zero-copy access
    // ------------------------------------------------------------------

    /// Get the raw bytes of this tensor (zero-copy reference).
    ///
    /// The bytes are in the tensor's native format (e.g., little-endian f32).
    /// This is the key to zero-copy transport: send these bytes directly
    /// over the wire, no serialization needed.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Create a tensor from raw bytes (zero-copy when possible).
    ///
    /// The caller is responsible for ensuring `bytes` contains valid data
    /// for the given dtype and shape.
    pub fn from_raw_bytes(dtype: Dtype, shape: Shape, bytes: Vec<u8>) -> Self {
        Self { dtype, shape, data: bytes }
    }

    /// Serialize this tensor to a compact binary wire format.
    ///
    /// Wire layout:
    /// - 1 byte:  dtype tag (0=F16, 1=F32, 2=F64, 3=I8, 4=I16, 5=I32, 6=I64, 7=Bool, 8=Ternary, 9=Utf8)
    /// - 2 bytes: ndims (u16 LE)
    /// - 8*N bytes: dimension sizes (u64 LE each; u64::MAX = Dynamic)
    /// - 8 bytes: data length in bytes (u64 LE)
    /// - var bytes: raw tensor data
    pub fn to_wire_bytes(&self) -> Vec<u8> {
        let dtype_tag: u8 = match self.dtype {
            Dtype::F16 => 0,
            Dtype::F32 => 1,
            Dtype::F64 => 2,
            Dtype::I8 => 3,
            Dtype::I16 => 4,
            Dtype::I32 => 5,
            Dtype::I64 => 6,
            Dtype::Bool => 7,
            Dtype::Ternary => 8,
            Dtype::Utf8 => 9,
        };

        let ndims = self.shape.0.len() as u16;
        let data_len = self.data.len() as u64;

        let header_size = 1 + 2 + (8 * ndims as usize) + 8;
        let mut buf = Vec::with_capacity(header_size + self.data.len());

        buf.push(dtype_tag);
        buf.extend_from_slice(&ndims.to_le_bytes());
        for dim in &self.shape.0 {
            let val: u64 = match dim {
                Dim::Fixed(n) => *n as u64,
                Dim::Dynamic => u64::MAX,
            };
            buf.extend_from_slice(&val.to_le_bytes());
        }
        buf.extend_from_slice(&data_len.to_le_bytes());
        buf.extend_from_slice(&self.data);

        buf
    }

    /// Deserialize a tensor from compact binary wire format.
    ///
    /// Returns `None` if the data is malformed.
    pub fn from_wire_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 3 {
            return None;
        }

        let dtype_tag = bytes[0];
        let dtype = match dtype_tag {
            0 => Dtype::F16,
            1 => Dtype::F32,
            2 => Dtype::F64,
            3 => Dtype::I8,
            4 => Dtype::I16,
            5 => Dtype::I32,
            6 => Dtype::I64,
            7 => Dtype::Bool,
            8 => Dtype::Ternary,
            9 => Dtype::Utf8,
            _ => return None,
        };

        let ndims = u16::from_le_bytes([bytes[1], bytes[2]]) as usize;

        let dims_end = 3 + ndims * 8;
        if bytes.len() < dims_end + 8 {
            return None;
        }

        let mut dims = Vec::with_capacity(ndims);
        for i in 0..ndims {
            let offset = 3 + i * 8;
            let val = u64::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
                bytes[offset + 4],
                bytes[offset + 5],
                bytes[offset + 6],
                bytes[offset + 7],
            ]);
            if val == u64::MAX {
                dims.push(Dim::Dynamic);
            } else {
                dims.push(Dim::Fixed(val as usize));
            }
        }

        let data_len_offset = dims_end;
        let data_len = u64::from_le_bytes([
            bytes[data_len_offset],
            bytes[data_len_offset + 1],
            bytes[data_len_offset + 2],
            bytes[data_len_offset + 3],
            bytes[data_len_offset + 4],
            bytes[data_len_offset + 5],
            bytes[data_len_offset + 6],
            bytes[data_len_offset + 7],
        ]) as usize;

        let data_start = data_len_offset + 8;
        if bytes.len() < data_start + data_len {
            return None;
        }

        let data = bytes[data_start..data_start + data_len].to_vec();

        Some(TensorData {
            dtype,
            shape: Shape(dims),
            data,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_has_one_element() {
        let s = Shape::scalar();
        assert_eq!(s.numel(), Some(1));
        assert_eq!(s.rank(), 0);
    }

    #[test]
    fn matrix_shape() {
        let s = Shape::matrix(28, 28);
        assert_eq!(s.numel(), Some(784));
        assert_eq!(s.rank(), 2);
    }

    #[test]
    fn tensor_type_display() {
        let t = TensorType::f32_matrix(28, 28);
        assert_eq!(t.to_string(), "Tensor<f32>[28, 28]");
    }

    #[test]
    fn ternary_size() {
        let t = TensorType::ternary_matrix(1024, 1024);
        assert_eq!(t.size_bytes(), Some(1024 * 1024)); // 1 byte per ternary
    }

    #[test]
    fn f32_roundtrip() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = TensorData::from_f32(Shape::vector(4), &values);
        assert_eq!(tensor.as_f32_slice().unwrap(), values);
    }

    #[test]
    fn shape_compatibility() {
        let a = Shape::matrix(28, 28);
        let b = Shape::matrix(28, 28);
        assert!(a.is_compatible_with(&b));

        let c = Shape::matrix(28, 14);
        assert!(!a.is_compatible_with(&c));
    }

    // ---- Zero-copy access tests ----

    #[test]
    fn tensor_as_bytes_is_zero_copy() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = TensorData::from_f32(Shape::vector(4), &values);

        let bytes = tensor.as_bytes();
        assert_eq!(bytes.len(), 16); // 4 floats * 4 bytes

        // Verify the bytes are the raw f32 little-endian representation
        let first_float = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(first_float, 1.0);
    }

    #[test]
    fn tensor_from_raw_bytes() {
        let original = TensorData::from_f32(Shape::matrix(2, 2), &[1.0, 2.0, 3.0, 4.0]);
        let bytes = original.as_bytes().to_vec();

        let reconstructed = TensorData::from_raw_bytes(Dtype::F32, Shape::matrix(2, 2), bytes);
        assert_eq!(reconstructed.as_f32_slice().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    // ---- Wire format tests ----

    #[test]
    fn tensor_wire_format_roundtrip() {
        let tensor = TensorData::from_f32(Shape::matrix(3, 2), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let wire = tensor.to_wire_bytes();

        let decoded = TensorData::from_wire_bytes(&wire).unwrap();
        assert_eq!(decoded.dtype, Dtype::F32);
        assert_eq!(decoded.shape, Shape::matrix(3, 2));
        assert_eq!(decoded.as_f32_slice().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn tensor_wire_format_preserves_precision() {
        // This is the key advantage over JSON: no float-to-string precision loss
        let hard_float = std::f32::consts::PI;
        let tensor = TensorData::from_f32(Shape::scalar(), &[hard_float]);
        let wire = tensor.to_wire_bytes();
        let decoded = TensorData::from_wire_bytes(&wire).unwrap();
        let result = decoded.as_f32_slice().unwrap()[0];
        assert_eq!(result, hard_float); // Exact equality, not approximate!
    }

    #[test]
    fn tensor_wire_vs_json_size() {
        // Demonstrate size savings: binary vs JSON
        let values: Vec<f32> = (0..768).map(|i| (i as f32) * 0.001).collect();
        let tensor = TensorData::from_f32(Shape::vector(768), &values);

        let wire_bytes = tensor.to_wire_bytes();
        let json_bytes = serde_json::to_vec(&tensor).unwrap();

        // Binary should be much smaller than JSON
        // 768 * 4 = 3072 bytes data + small header vs JSON with text floats
        assert!(wire_bytes.len() < json_bytes.len(),
            "wire={} should be < json={}", wire_bytes.len(), json_bytes.len());
    }

    #[test]
    fn tensor_wire_format_scalar() {
        let tensor = TensorData::from_f32(Shape::scalar(), &[42.0]);
        let wire = tensor.to_wire_bytes();
        let decoded = TensorData::from_wire_bytes(&wire).unwrap();
        assert_eq!(decoded.shape, Shape::scalar());
        assert_eq!(decoded.as_f32_slice().unwrap(), vec![42.0]);
    }

    #[test]
    fn tensor_wire_format_rejects_truncated() {
        // Too short: no dtype even
        assert!(TensorData::from_wire_bytes(&[]).is_none());
        // Too short: dtype + partial ndims
        assert!(TensorData::from_wire_bytes(&[1, 0]).is_none());
        // Claims 1 dim but no dim data follows
        assert!(TensorData::from_wire_bytes(&[1, 1, 0]).is_none());
    }
}
