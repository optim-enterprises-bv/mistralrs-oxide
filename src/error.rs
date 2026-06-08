use thiserror::Error;

#[derive(Error, Debug)]
pub enum OxideError {
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },

    #[error("Dtype mismatch: expected {expected}, got {got}")]
    DtypeMismatch { expected: String, got: String },

    #[error("Device mismatch: {0}")]
    DeviceMismatch(String),

    #[error("CUDA error: {0}")]
    CudaError(String),

    #[error("Device error: {0}")]
    DeviceError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("Out of memory: requested {requested} bytes, available {available} bytes")]
    OutOfMemory { requested: usize, available: usize },

    #[error("Kernel compilation error: {0}")]
    KernelCompileError(String),

    #[error("Kernel launch error: {0}")]
    KernelLaunchError(String),

    #[error("Not implemented: {0}")]
    NotImplemented(String),
}

pub type OxideResult<T> = Result<T, OxideError>;

#[macro_export]
macro_rules! bail {
    ($($arg:tt)*) => {
        return Err($crate::error::OxideError::InvalidArgument(format!($($arg)*)))
    };
}

#[macro_export]
macro_rules! unsupported {
    ($($arg:tt)*) => {
        return Err($crate::error::OxideError::UnsupportedOperation(format!($($arg)*)))
    };
}
