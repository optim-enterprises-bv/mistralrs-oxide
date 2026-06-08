use std::fmt::{self, Display};
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    U8,
    U32,
    I64,
    BF16,
    F16,
    F32,
    F64,
    F8E4M3,
}

impl DType {
    pub fn size_in_bytes(&self) -> usize {
        match self {
            DType::U8 | DType::F8E4M3 => 1,
            DType::U32 => 4,
            DType::I64 => 8,
            DType::BF16 | DType::F16 => 2,
            DType::F32 => 4,
            DType::F64 => 8,
        }
    }

    pub fn is_float(&self) -> bool {
        matches!(self, DType::F16 | DType::BF16 | DType::F32 | DType::F64 | DType::F8E4M3)
    }

    pub fn is_int(&self) -> bool {
        matches!(self, DType::U8 | DType::U32 | DType::I64)
    }
}

impl Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::U8 => write!(f, "u8"),
            DType::U32 => write!(f, "u32"),
            DType::I64 => write!(f, "i64"),
            DType::BF16 => write!(f, "bf16"),
            DType::F16 => write!(f, "f16"),
            DType::F32 => write!(f, "f32"),
            DType::F64 => write!(f, "f64"),
            DType::F8E4M3 => write!(f, "f8e4m3"),
        }
    }
}

impl FromStr for DType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "u8" => Ok(DType::U8),
            "u32" => Ok(DType::U32),
            "i64" => Ok(DType::I64),
            "bf16" => Ok(DType::BF16),
            "f16" => Ok(DType::F16),
            "f32" => Ok(DType::F32),
            "f64" => Ok(DType::F64),
            "f8e4m3" => Ok(DType::F8E4M3),
            _ => Err(format!("Unknown dtype: {}", s)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: [usize; 8],
    rank: usize,
}

impl Shape {
    pub fn new(dims: &[usize]) -> Self {
        let rank = dims.len();
        assert!(rank <= 8, "Shape rank limited to 8 dimensions");
        let mut arr = [1usize; 8];
        arr[..rank].copy_from_slice(dims);
        Self { dims: arr, rank }
    }

    pub fn scalar() -> Self {
        Self { dims: [1; 8], rank: 0 }
    }

    pub fn from_dims(dims: &[usize]) -> Self {
        Self::new(dims)
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims[..self.rank]
    }

    pub fn elem_count(&self) -> usize {
        self.dims[..self.rank].iter().product()
    }

    pub fn stride_contiguous(&self) -> Vec<usize> {
        let mut stride = vec![0usize; self.rank];
        let mut current = 1;
        for i in (0..self.rank).rev() {
            stride[i] = current;
            current *= self.dims[i];
        }
        stride
    }

    pub fn is_contiguous(&self, strides: &[usize]) -> bool {
        if strides.len() != self.rank {
            return false;
        }
        let expected = self.stride_contiguous();
        strides == expected
    }

    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Self {
        let mut new_dims = self.dims().to_vec();
        new_dims[dim] = len;
        Self::new(&new_dims)
    }
}

impl Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, dim) in self.dims[..self.rank].iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", dim)?;
        }
        write!(f, ")")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
    pub shape: Shape,
    pub strides: [usize; 8],
    pub offset: usize,
}

impl Layout {
    pub fn new(shape: Shape, strides: Vec<usize>, offset: usize) -> Self {
        let mut arr = [0usize; 8];
        arr[..shape.rank()].copy_from_slice(&strides);
        Self {
            shape,
            strides: arr,
            offset,
        }
    }

    pub fn contiguous(shape: Shape) -> Self {
        let strides = shape.stride_contiguous();
        Self::new(shape, strides, 0)
    }

    pub fn is_contiguous(&self) -> bool {
        self.shape.is_contiguous(&self.strides[..self.shape.rank()])
    }

    pub fn num_elements(&self) -> usize {
        self.shape.elem_count()
    }

    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Self {
        let new_shape = self.shape.narrow(dim, start, len);
        let mut new_layout = self.clone();
        new_layout.shape = new_shape;
        new_layout.offset += start * self.strides[dim];
        new_layout
    }
}

impl Clone for Layout {
    fn clone(&self) -> Self {
        Self {
            shape: self.shape,
            strides: self.strides,
            offset: self.offset,
        }
    }
}
