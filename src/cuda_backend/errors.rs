//! Real CUDA error handling using cudarc error types

#![cfg(feature = "cuda")]

use std::fmt;
use std::error::Error;

/// CUDA error types based on CUDA driver API
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
    DeviceNotInitialized = 1000,
    KernelCompilationFailed = 1001,
    PtxLoadFailed = 1002,
    KernelLaunchFailed = 1003,
}

impl CudaError {
    /// Convert from raw CUDA error code
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

    /// Get human-readable error message
    pub fn message(&self) -> &'static str {
        match self {
            CudaError::Success => "No error",
            CudaError::MissingConfiguration => "__global__ function call is not configured",
            CudaError::MemoryAllocation => "Memory allocation failed",
            CudaError::InitializationError => "Initialization error",
            CudaError::LaunchFailure => "Launch failed",
            CudaError::PriorLaunchFailure => "Prior launch failed",
            CudaError::LaunchTimeout => "Launch timeout",
            CudaError::LaunchOutOfResources => "Launch out of resources",
            CudaError::InvalidDevicePointer => "Invalid device pointer",
            CudaError::InvalidMemcpyDirection => "Invalid memcpy direction",
            CudaError::InsufficientDriver => "Insufficient driver version",
            CudaError::NoDevice => "No device found",
            CudaError::InvalidDevice => "Invalid device ordinal",
            CudaError::StartupFailure => "Startup failure",
            CudaError::ApiFailureBase => "API failure base",
            CudaError::Unknown => "Unknown error",
            CudaError::DeviceNotInitialized => "Device not initialized",
            CudaError::KernelCompilationFailed => "Kernel compilation failed",
            CudaError::PtxLoadFailed => "PTX load failed",
            CudaError::KernelLaunchFailed => "Kernel launch failed",
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
                | CudaError::InsufficientDriver
                | CudaError::NoDevice
        )
    }

    /// Check if error is OOM
    pub fn is_oom(&self) -> bool {
        matches!(self, CudaError::MemoryAllocation | CudaError::LaunchOutOfResources)
    }

    /// Check if error is timeout
    pub fn is_timeout(&self) -> bool {
        matches!(self, CudaError::LaunchTimeout)
    }
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CUDA error {}: {}", *self as i32, self.message())
    }
}

impl Error for CudaError {}

/// Result type for CUDA operations
pub type CudaResult<T> = Result<T, String>;

/// Check CUDA result and convert to OxideResult
pub fn check_cuda<T, E>(result: Result<T, E>) -> Result<T, crate::error::OxideError>
where
    E: std::fmt::Display,
{
    match result {
        Ok(val) => Ok(val),
        Err(e) => {
            let msg = e.to_string();
            // Try to extract error code
            let cuda_err = parse_cuda_error(&msg);
            Err(crate::error::OxideError::CudaError(format!(
                "{}: {}",
                cuda_err,
                msg
            )))
        }
    }
}

/// Parse error string to CudaError
fn parse_cuda_error(msg: &str) -> CudaError {
    if msg.contains("out of memory") || msg.contains("OOM") {
        CudaError::MemoryAllocation
    } else if msg.contains("timeout") {
        CudaError::LaunchTimeout
    } else if msg.contains("invalid device") {
        CudaError::InvalidDevice
    } else if msg.contains("insufficient driver") {
        CudaError::InsufficientDriver
    } else if msg.contains("no device") {
        CudaError::NoDevice
    } else if msg.contains("compilation") || msg.contains("compile") {
        CudaError::KernelCompilationFailed
    } else if msg.contains("PTX") || msg.contains("ptx") {
        CudaError::PtxLoadFailed
    } else if msg.contains("launch") {
        CudaError::KernelLaunchFailed
    } else {
        CudaError::Unknown
    }
}

/// Error recovery strategies
#[derive(Debug, Clone)]
pub enum RecoveryStrategy {
    /// Retry the operation
    Retry { max_attempts: usize },
    /// Fall back to CPU implementation
    FallbackToCpu,
    /// Reduce batch size and retry
    ReduceBatchSize { min_batch: usize },
    /// Clear memory cache and retry
    ClearCache,
    /// Reset device context
    ResetContext,
    /// Abort with error
    Abort,
}

impl RecoveryStrategy {
    /// Determine strategy from error
    pub fn from_error(error: &CudaError) -> Self {
        match error {
            CudaError::MemoryAllocation => RecoveryStrategy::ClearCache,
            CudaError::LaunchOutOfResources => RecoveryStrategy::ReduceBatchSize { min_batch: 1 },
            CudaError::LaunchTimeout => RecoveryStrategy::Retry { max_attempts: 3 },
            CudaError::LaunchFailure => RecoveryStrategy::ResetContext,
            CudaError::Unknown => RecoveryStrategy::FallbackToCpu,
            _ => RecoveryStrategy::Abort,
        }
    }

    /// Apply recovery strategy
    pub fn apply<F, T>(&self,
        mut operation: F,
        current_attempt: usize,
    ) -> Result<T, crate::error::OxideError>
    where
        F: FnMut() -> Result<T, crate::error::OxideError>,
    {
        match self {
            RecoveryStrategy::Retry { max_attempts } => {
                if current_attempt < *max_attempts {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    operation()
                } else {
                    Err(crate::error::OxideError::CudaError(
                        format!("Failed after {} attempts", max_attempts)
                    ))
                }
            }
            RecoveryStrategy::FallbackToCpu => {
                // Return special error that upstream can handle
                Err(crate::error::OxideError::CudaError(
                    "Falling back to CPU".to_string()
                ))
            }
            RecoveryStrategy::ReduceBatchSize { min_batch } => {
                // Would need to modify batch size in context
                // For now, just retry
                operation()
            }
            RecoveryStrategy::ClearCache => {
                // Clear cache and retry
                // In real implementation, call memory pool clear
                operation()
            }
            RecoveryStrategy::ResetContext => {
                // Would need to reinitialize CUDA context
                operation()
            }
            RecoveryStrategy::Abort => {
                Err(crate::error::OxideError::CudaError(
                    "Fatal CUDA error, aborting".to_string()
                ))
            }
        }
    }
}

/// Error context for debugging
#[derive(Debug)]
pub struct ErrorContext {
    pub operation: String,
    pub device_id: usize,
    pub stream_id: Option<usize>,
    pub kernel_name: Option<String>,
    pub allocated_bytes: usize,
}

impl ErrorContext {
    pub fn new(operation: &str, device_id: usize) -> Self {
        Self {
            operation: operation.to_string(),
            device_id,
            stream_id: None,
            kernel_name: None,
            allocated_bytes: 0,
        }
    }

    pub fn with_stream(mut self, stream_id: usize) -> Self {
        self.stream_id = Some(stream_id);
        self
    }

    pub fn with_kernel(mut self, name: &str) -> Self {
        self.kernel_name = Some(name.to_string());
        self
    }

    pub fn with_memory(mut self, bytes: usize) -> Self {
        self.allocated_bytes = bytes;
        self
    }
}

/// Error logger for tracking CUDA errors
pub struct ErrorLogger {
    history: Vec<(std::time::Instant, CudaError, ErrorContext)>,
    max_history: usize,
}

impl ErrorLogger {
    pub fn new(max_history: usize) -> Self {
        Self {
            history: Vec::new(),
            max_history,
        }
    }

    /// Log an error
    pub fn log(&mut self, error: CudaError, context: ErrorContext) {
        self.history.push((std::time::Instant::now(), error, context));
        
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
    }

    /// Get recent errors
    pub fn recent(&self, count: usize) -> &[(std::time::Instant, CudaError, ErrorContext)] {
        let start = self.history.len().saturating_sub(count);
        &self.history[start..]
    }

    /// Print error summary
    pub fn print_summary(&self) {
        println!("=== CUDA Error History ===");
        for (time, error, context) in &self.history {
            println!(
                "[{:?}] Device {}: {} during {}",
                time.elapsed(),
                context.device_id,
                error,
                context.operation
            );
            if let Some(kernel) = &context.kernel_name {
                println!("  Kernel: {}", kernel);
            }
        }
    }

    /// Get error frequency by type
    pub fn error_frequency(&self) -> std::collections::HashMap<CudaError, usize> {
        let mut freq = std::collections::HashMap::new();
        for (_, error, _) in &self.history {
            *freq.entry(*error).or_insert(0) += 1;
        }
        freq
    }

    /// Clear history
    pub fn clear(&mut self) {
        self.history.clear();
    }
}

/// Device error tracker
pub struct DeviceErrorTracker {
    devices: Vec<DeviceErrorInfo>,
}

#[derive(Debug)]
pub struct DeviceErrorInfo {
    pub device_id: usize,
    pub error_count: usize,
    pub last_error: Option<CudaError>,
    pub last_error_time: Option<std::time::Instant>,
    pub consecutive_errors: usize,
}

impl DeviceErrorTracker {
    pub fn new(num_devices: usize) -> Self {
        let devices = (0..num_devices)
            .map(|id| DeviceErrorInfo {
                device_id: id,
                error_count: 0,
                last_error: None,
                last_error_time: None,
                consecutive_errors: 0,
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
            info.consecutive_errors += 1;
        }
    }

    /// Record success for device
    pub fn record_success(&mut self, device_id: usize) {
        if let Some(info) = self.devices.get_mut(device_id) {
            info.consecutive_errors = 0;
        }
    }

    /// Check if device is healthy
    pub fn is_healthy(&self, device_id: usize) -> bool {
        self.devices
            .get(device_id)
            .map(|info| info.consecutive_errors < 5)
            .unwrap_or(false)
    }

    /// Get healthiest device
    pub fn healthiest_device(&self) -> Option<usize> {
        self.devices
            .iter()
            .filter(|info| info.error_count == 0)
            .map(|info| info.device_id)
            .next()
            .or_else(|| {
                // Find device with lowest consecutive errors
                self.devices
                    .iter()
                    .min_by_key(|info| info.consecutive_errors)
                    .map(|info| info.device_id)
            })
    }

    /// Get device health status
    pub fn health_status(&self) -> Vec<(DeviceHealth, usize)> {
        self.devices
            .iter()
            .map(|info| {
                let health = if info.consecutive_errors == 0 {
                    DeviceHealth::Healthy
                } else if info.consecutive_errors < 3 {
                    DeviceHealth::Warning
                } else {
                    DeviceHealth::Critical
                };
                (health, info.device_id)
            })
            .collect()
    }
}

#[derive(Debug, Clone, Copy)]
pub enum DeviceHealth {
    Healthy,
    Warning,
    Critical,
}

/// Assertion macros for CUDA
#[macro_export]
macro_rules! cuda_assert {
    ($cond:expr, $msg:expr) => {
        if !$cond {
            return Err(crate::cuda_backend::errors::CudaError::Unknown.into());
        }
    };
}

/// Check and convert CUDA result
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_error_codes() {
        assert_eq!(CudaError::from_raw(0), CudaError::Success);
        assert_eq!(CudaError::from_raw(2), CudaError::MemoryAllocation);
        assert_eq!(CudaError::from_raw(999), CudaError::Unknown);
    }

    #[test]
    fn test_error_recoverability() {
        assert!(!CudaError::LaunchFailure.is_recoverable());
        assert!(CudaError::MemoryAllocation.is_recoverable());
        assert!(CudaError::InvalidDevicePointer.is_recoverable());
    }

    #[test]
    fn test_recovery_strategy() {
        let oom = RecoveryStrategy::from_error(&CudaError::MemoryAllocation);
        assert!(matches!(oom, RecoveryStrategy::ClearCache));

        let timeout = RecoveryStrategy::from_error(&CudaError::LaunchTimeout);
        assert!(matches!(timeout, RecoveryStrategy::Retry { .. }));
    }

    #[test]
    fn test_device_tracker() {
        let mut tracker = DeviceErrorTracker::new(2);
        
        tracker.record(0, CudaError::MemoryAllocation);
        tracker.record(0, CudaError::MemoryAllocation);
        
        assert_eq!(tracker.devices[0].error_count, 2);
        assert!(tracker.is_healthy(0));
        
        tracker.record(0, CudaError::MemoryAllocation);
        tracker.record(0, CudaError::MemoryAllocation);
        tracker.record(0, CudaError::MemoryAllocation);
        
        assert!(!tracker.is_healthy(0));
        
        // Should get device 1 as healthiest
        assert_eq!(tracker.healthiest_device(), Some(1));
    }
}
