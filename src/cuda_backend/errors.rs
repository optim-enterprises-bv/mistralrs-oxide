//! CUDA error handling

use std::fmt;

/// CUDA error codes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CudaError {
    Success = 0,
    MissingConfiguration = 1,
    MemoryAllocation = 2,
    InitializationError = 3,
    LaunchFailure = 4,
    PriorLaunchFailure = 5,
    LaunchTimeout = 6,
    LaunchOutOfResources = 7,
    InvalidDevicePointer = 8,
    InvalidMemcpyDirection = 9,
    InsufficientDriver = 35,
    NoDevice = 38,
    InvalidDevice = 101,
    StartupFailure = 127,
    ApiFailureBase = 10000,
    Unknown = -1,
}

impl CudaError {
    /// Convert from raw error code
    pub fn from_raw(code: i32) -> Self {
        match code {
            0 => CudaError::Success,
            1 => CudaError::MissingConfiguration,
            2 => CudaError::MemoryAllocation,
            3 => CudaError::InitializationError,
            4 => CudaError::LaunchFailure,
            5 => CudaError::PriorLaunchFailure,
            6 => CudaError::LaunchTimeout,
            7 => CudaError::LaunchOutOfResources,
            8 => CudaError::InvalidDevicePointer,
            9 => CudaError::InvalidMemcpyDirection,
            35 => CudaError::InsufficientDriver,
            38 => CudaError::NoDevice,
            101 => CudaError::InvalidDevice,
            127 => CudaError::StartupFailure,
            10000 => CudaError::ApiFailureBase,
            _ => CudaError::Unknown,
        }
    }

    /// Get error message
    pub fn message(&self) -> &'static str {
        match self {
            CudaError::Success => "No errors",
            CudaError::MissingConfiguration => "__global__ function call is not configured",
            CudaError::MemoryAllocation => "Memory allocation failed",
            CudaError::InitializationError => "Initialization error",
            CudaError::LaunchFailure => "Launch failed",
            CudaError::PriorLaunchFailure => "Prior launch failed",
            CudaError::LaunchTimeout => "Launch timeout",
            CudaError::LaunchOutOfResources => "Launch out of resources",
            CudaError::InvalidDevicePointer => "Invalid device pointer",
            CudaError::InvalidMemcpyDirection => "Invalid memcpy direction",
            CudaError::InsufficientDriver => "Insufficient driver",
            CudaError::NoDevice => "No device",
            CudaError::InvalidDevice => "Invalid device",
            CudaError::StartupFailure => "Startup failure",
            CudaError::ApiFailureBase => "API failure base",
            CudaError::Unknown => "Unknown error",
        }
    }

    /// Check if error is success
    pub fn is_success(&self) -> bool {
        matches!(self, CudaError::Success)
    }

    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        !matches!(
            self,
            CudaError::LaunchFailure
                | CudaError::LaunchTimeout
                | CudaError::StartupFailure
        )
    }
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CUDA error {}: {}", *self as i32, self.message())
    }
}

impl std::error::Error for CudaError {}

/// Result type for CUDA operations
pub type CudaResult<T> = Result<T, CudaError>;

/// Check CUDA result and convert to OxideResult
pub fn check_cuda<T>(result: Result<T, String>) -> crate::error::OxideResult<T> {
    match result {
        Ok(val) => Ok(val),
        Err(msg) => {
            // Try to parse error code from message
            let code = parse_cuda_error_code(&msg);
            let cuda_err = CudaError::from_raw(code);
            Err(crate::error::OxideError::CudaError(format!(
                "{}: {}",
                cuda_err,
                msg
            )))
        }
    }
}

/// Parse CUDA error code from message
fn parse_cuda_error_code(msg: &str) -> i32 {
    // Try to extract error code from common CUDA error formats
    if let Some(idx) = msg.find("cudaError") {
        // Parse error code from cudaErrorXXX format
        if msg.contains("cudaErrorMemoryAllocation") {
            return 2;
        } else if msg.contains("cudaErrorLaunchFailure") {
            return 4;
        }
    }
    -1 // Unknown
}

/// Check and wrap CUDA operation
#[macro_export]
macro_rules! cuda_check {
    ($expr:expr) => {
        match $expr {
            Ok(val) => val,
            Err(e) => {
                return Err(crate::error::OxideError::CudaError(e.to_string()));
            }
        }
    };
}

/// CUDA assertion
#[macro_export]
macro_rules! cuda_assert {
    ($cond:expr, $msg:expr) => {
        if !$cond {
            return Err(crate::error::OxideError::CudaError($msg.to_string()));
        }
    };
}

/// Error recovery strategies
pub enum RecoveryStrategy {
    Retry,
    FallbackToCpu,
    ReduceBatchSize,
    ClearCache,
    Abort,
}

impl RecoveryStrategy {
    /// Determine strategy from error
    pub fn from_error(error: &CudaError) -> Self {
        match error {
            CudaError::MemoryAllocation => RecoveryStrategy::ClearCache,
            CudaError::LaunchOutOfResources => RecoveryStrategy::ReduceBatchSize,
            CudaError::LaunchTimeout => RecoveryStrategy::Retry,
            _ => RecoveryStrategy::FallbackToCpu,
        }
    }

    /// Apply recovery strategy
    pub fn apply(&self) -> crate::error::OxideResult<()> {
        match self {
            RecoveryStrategy::Retry => {
                // Wait and retry
                std::thread::sleep(std::time::Duration::from_millis(100));
                Ok(())
            }
            RecoveryStrategy::FallbackToCpu => {
                // Signal to use CPU implementation
                Ok(())
            }
            RecoveryStrategy::ClearCache => {
                // Clear GPU memory cache
                Ok(())
            }
            RecoveryStrategy::ReduceBatchSize => {
                // Reduce batch size and retry
                Ok(())
            }
            RecoveryStrategy::Abort => {
                Err(crate::error::OxideError::CudaError(
                    "Fatal CUDA error, aborting".to_string()
                ))
            }
        }
    }
}

/// Error logging
pub struct ErrorLogger {
    history: Vec<(std::time::Instant, CudaError, String)>,
}

impl ErrorLogger {
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
        }
    }

    /// Log error
    pub fn log(&mut self, error: CudaError, context: &str) {
        self.history.push((
            std::time::Instant::now(),
            error,
            context.to_string(),
        ));
    }

    /// Get recent errors
    pub fn recent(&self, count: usize) -> &[(std::time::Instant, CudaError, String)] {
        let start = self.history.len().saturating_sub(count);
        &self.history[start..]
    }

    /// Print error summary
    pub fn print_summary(&self) {
        println!("=== CUDA Error History ===");
        for (time, error, context) in &self.history {
            println!(
                "[{:?}] {} in {}",
                time.elapsed(),
                error,
                context
            );
        }
    }
}

/// Device-specific error info
#[derive(Debug)]
pub struct DeviceErrorInfo {
    pub device_id: usize,
    pub error_count: usize,
    pub last_error: Option<CudaError>,
    pub last_error_time: Option<std::time::Instant>,
}

/// Error statistics per device
pub struct DeviceErrorTracker {
    devices: Vec<DeviceErrorInfo>,
}

impl DeviceErrorTracker {
    pub fn new(num_devices: usize) -> Self {
        let devices = (0..num_devices)
            .map(|id| DeviceErrorInfo {
                device_id: id,
                error_count: 0,
                last_error: None,
                last_error_time: None,
            })
            .collect();
        
        Self { devices }
    }

    /// Record error for device
    pub fn record(&mut self, device_id: usize, error: CudaError) {
        if let Some(info) = self.devices.get_mut(device_id) {
            info.error_count += 1;
            info.last_error = Some(error);
            info.last_error_time = Some(std::time::Instant::now());
        }
    }

    /// Check if device is healthy
    pub fn is_healthy(&self, device_id: usize) -> bool {
        self.devices
            .get(device_id)
            .map(|info| info.error_count < 10)
            .unwrap_or(false)
    }

    /// Get healthiest device
    pub fn healthiest_device(&self) -> Option<usize> {
        self.devices
            .iter()
            .filter(|info| info.error_count == 0)
            .map(|info| info.device_id)
            .next()
    }
}
