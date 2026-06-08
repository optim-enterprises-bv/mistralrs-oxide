//! CUDA backend using cudarc - production-ready GPU acceleration
//!
//! This module provides actual GPU execution via the cudarc crate,
//! which wraps CUDA driver APIs in safe Rust.

#![cfg(feature = "cuda")]

use std::sync::Arc;
use cudarc::driver::{CudaContext, CudaDevice, CudaStream, DeviceSlice, DevicePtr, LaunchConfig};
use crate::core::{Device, DType, Shape, Tensor};
use crate::error::{OxideError, OxideResult};

pub mod memory;
pub mod streams;
pub mod launcher;
pub mod cublas;
pub mod errors;
pub mod context;

pub use memory::{CudaMemoryPool, DeviceBufferExt, PinnedBuffer};
pub use streams::{CudaStreamPool, StreamHandle, EventHandle};
pub use launcher::{KernelLauncher, KernelModule, GridConfig, BlockConfig};
pub use cublas::{CublasContext, GemmConfig};
pub use errors::{CudaResult, check_cuda_error, CudaError};
pub use context::{CudaContextManager, DeviceProperties};

/// CUDA backend for tensor operations
pub struct CudaBackend {
    device: Arc<CudaDevice>,
    context: Arc<CudaContext>,
    memory_pool: CudaMemoryPool,
    stream_pool: CudaStreamPool,
    cublas: Option<CublasContext>,
    device_id: usize,
}

impl CudaBackend {
    /// Initialize CUDA backend for specified device
    pub fn new(device_id: usize) -> OxideResult<Self> {
        // Get or create CUDA device
        let device = CudaDevice::new(device_id)
            .map_err(|e| OxideError::CudaError(format!(
                "Failed to initialize CUDA device {}: {}", device_id, e
            )))?;
        
        let device = Arc::new(device);
        let context = Arc::new(device.default_context());
        
        // Initialize memory pool
        let memory_pool = CudaMemoryPool::new(&device)?;
        
        // Initialize stream pool
        let stream_pool = CudaStreamPool::new(&device, 8)?;
        
        // Try to initialize cuBLAS
        let cublas = CublasContext::new(&device).ok();
        
        Ok(Self {
            device,
            context,
            memory_pool,
            stream_pool,
            cublas,
            device_id,
        })
    }

    /// Get CUDA device
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Get default stream
    pub fn default_stream(&self) -> &CudaStream {
        self.device.default_stream()
    }

    /// Acquire stream from pool
    pub fn acquire_stream(&self) -> OxideResult<StreamHandle> {
        self.stream_pool.acquire()
    }

    /// Get memory pool
    pub fn memory_pool(&self) -> &CudaMemoryPool {
        &self.memory_pool
    }

    /// Get cuBLAS context
    pub fn cublas(&self) -> Option<&CublasContext> {
        self.cublas.as_ref()
    }

    /// Get device properties
    pub fn properties(&self) -> DeviceProperties {
        DeviceProperties::from_device(&self.device, self.device_id)
    }

    /// Synchronize all streams
    pub fn synchronize(&self) -> OxideResult<()> {
        self.device.synchronize()
            .map_err(|e| OxideError::CudaError(format!("Synchronize failed: {}", e)))
    }

    /// Get available memory
    pub fn memory_info(&self) -> (usize, usize) {
        // (free, total)
        self.device.memory_info()
    }
}

/// Global CUDA backend manager
pub struct CudaBackendManager {
    backends: Vec<Option<CudaBackend>>,
}

impl CudaBackendManager {
    /// Initialize all available CUDA devices
    pub fn initialize_all() -> OxideResult<Self> {
        let device_count = cudarc::driver::device_count()
            .map_err(|e| OxideError::CudaError(format!("Failed to get device count: {}", e)))?;
        
        let mut backends = Vec::with_capacity(device_count);
        
        for i in 0..device_count {
            match CudaBackend::new(i) {
                Ok(backend) => {
                    let props = backend.properties();
                    println!("Initialized CUDA device {}: {} (CC {}.{}, {} MB)",
                        i, props.name, props.major, props.minor, props.total_memory / (1024 * 1024));
                    backends.push(Some(backend));
                }
                Err(e) => {
                    eprintln!("Failed to initialize CUDA device {}: {}", i, e);
                    backends.push(None);
                }
            }
        }
        
        if backends.iter().all(|b| b.is_none()) {
            return Err(OxideError::CudaError(
                "No CUDA devices available".to_string()
            ));
        }
        
        Ok(Self { backends })
    }

    /// Get backend for device
    pub fn get(&self, device_id: usize) -> Option<&CudaBackend> {
        self.backends.get(device_id).and_then(|b| b.as_ref())
    }

    /// Get mutable backend for device
    pub fn get_mut(&mut self, device_id: usize) -> Option<&mut CudaBackend> {
        self.backends.get_mut(device_id).and_then(|b| b.as_mut())
    }

    /// Get any available backend
    pub fn get_any(&self) -> Option<&CudaBackend> {
        self.backends.iter().find_map(|b| b.as_ref())
    }

    /// Number of initialized devices
    pub fn num_devices(&self) -> usize {
        self.backends.iter().filter(|b| b.is_some()).count()
    }
}

/// Check if CUDA is available
pub fn is_cuda_available() -> bool {
    cudarc::driver::device_count().map(|c| c > 0).unwrap_or(false)
}

/// Get number of CUDA devices
pub fn device_count() -> usize {
    cudarc::driver::device_count().unwrap_or(0)
}

/// Get compute capability for device
pub fn compute_capability(device_id: usize) -> Option<(i32, i32)> {
    use cudarc::driver::CudaDevice;
    
    CudaDevice::new(device_id).ok().map(|dev| {
        let major = dev.attribute(cudarc::driver::DeviceAttribute::ComputeCapabilityMajor).unwrap_or(0);
        let minor = dev.attribute(cudarc::driver::DeviceAttribute::ComputeCapabilityMinor).unwrap_or(0);
        (major, minor)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cuda_availability() {
        let available = is_cuda_available();
        println!("CUDA available: {}", available);
        
        if available {
            let count = device_count();
            println!("Found {} CUDA device(s)", count);
            
            for i in 0..count.min(1) {
                if let Some(cc) = compute_capability(i) {
                    println!("Device {}: CC {}.{}", i, cc.0, cc.1);
                }
            }
        }
    }
    
    #[test]
    fn test_backend_initialization() -> OxideResult<()> {
        if !is_cuda_available() {
            println!("Skipping test - no CUDA available");
            return Ok(());
        }
        
        let manager = CudaBackendManager::initialize_all()?;
        assert!(manager.num_devices() > 0);
        
        if let Some(backend) = manager.get_any() {
            let props = backend.properties();
            println!("Using device: {} ({} MB)", props.name, props.total_memory / (1024 * 1024));
        }
        
        Ok(())
    }
}
