//! CUDA backend integration using cuda-core and cuda-oxide
//! 
//! This module provides actual GPU execution capabilities when CUDA is available.

use std::sync::Arc;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig};
use crate::core::{Device, DType, Shape, Tensor, Storage, StorageData};
use crate::error::{OxideError, OxideResult, bail};

pub mod memory;
pub mod streams;
pub mod launcher;
pub mod cublas;
pub mod errors;

pub use memory::{GpuMemoryPool, DeviceBufferManager};
pub use streams::{StreamManager, GpuEvent};
pub use launcher::KernelLauncher;
pub use cublas::CuBLASHandle;
pub use errors::{check_cuda, CudaError};

/// CUDA backend state
pub struct CudaBackend {
    context: Arc<CudaContext>,
    device_id: usize,
    memory_pool: GpuMemoryPool,
    stream_manager: StreamManager,
    cublas: Option<CuBLASHandle>,
}

impl CudaBackend {
    /// Initialize CUDA backend for specified device
    pub fn new(device_id: usize) -> OxideResult<Self> {
        // Initialize CUDA context
        let context = CudaContext::new(device_id)
            .map_err(|e| OxideError::CudaError(format!("Failed to create CUDA context: {}", e)))?;
        
        let context = Arc::new(context);
        
        // Initialize memory pool
        let memory_pool = GpuMemoryPool::new(device_id)?;
        
        // Initialize stream manager
        let stream_manager = StreamManager::new(&context)?;
        
        // Optionally initialize cuBLAS
        let cublas = CuBLASHandle::new()
            .ok()
            .map(|h| h);
        
        Ok(Self {
            context,
            device_id,
            memory_pool,
            stream_manager,
            cublas,
        })
    }

    /// Get default stream
    pub fn default_stream(&self,
    ) -> &CudaStream {
        self.stream_manager.default_stream()
    }

    /// Create new stream for async operations
    pub fn create_stream(&self,
    ) -> OxideResult<CudaStream> {
        self.stream_manager.create_stream()
    }

    /// Get memory pool
    pub fn memory_pool(&self) -> &GpuMemoryPool {
        &self.memory_pool
    }

    /// Get cuBLAS handle if available
    pub fn cublas(&self) -> Option<&CuBLASHandle> {
        self.cublas.as_ref()
    }

    /// Synchronize all streams
    pub fn synchronize_all(&self) -> OxideResult<()> {
        self.stream_manager.synchronize_all()
    }

    /// Get device properties
    pub fn device_props(&self,
    ) -> DeviceProps {
        DeviceProps {
            device_id: self.device_id,
            name: format!("CUDA Device {}", self.device_id),
            compute_capability: (8, 6), // Placeholder
            total_memory: 24_000_000_000, // Placeholder: 24GB
            free_memory: 20_000_000_000,
        }
    }
}

/// Device properties
#[derive(Debug, Clone)]
pub struct DeviceProps {
    pub device_id: usize,
    pub name: String,
    pub compute_capability: (i32, i32),
    pub total_memory: usize,
    pub free_memory: usize,
}

impl DeviceProps {
    pub fn supports_tensor_cores(&self) -> bool {
        self.compute_capability.0 >= 7
    }
    
    pub fn supports_async_copy(&self) -> bool {
        self.compute_capability.0 >= 8
    }
}

/// Global CUDA backend manager
pub struct CudaBackendManager {
    backends: Vec<Option<CudaBackend>>,
}

impl CudaBackendManager {
    /// Initialize all available CUDA devices
    pub fn initialize_all() -> OxideResult<Self> {
        let device_count = get_cuda_device_count();
        let mut backends = Vec::with_capacity(device_count);
        
        for i in 0..device_count {
            match CudaBackend::new(i) {
                Ok(backend) => backends.push(Some(backend)),
                Err(e) => {
                    eprintln!("Failed to initialize CUDA device {}: {}", i, e);
                    backends.push(None);
                }
            }
        }
        
        Ok(Self { backends })
    }

    /// Get backend for device
    pub fn get(&self, device_id: usize) -> Option<&CudaBackend> {
        self.backends.get(device_id).and_then(|b| b.as_ref())
    }

    /// Get number of initialized devices
    pub fn num_devices(&self) -> usize {
        self.backends.iter().filter(|b| b.is_some()).count()
    }
}

/// Get CUDA device count
pub fn get_cuda_device_count() -> usize {
    // In real implementation, query CUDA driver
    // For now, return 1 assuming we have at least one GPU
    1
}

/// Check if CUDA is available
pub fn is_cuda_available() -> bool {
    get_cuda_device_count() > 0
}

/// Get CUDA compute capability
pub fn get_compute_capability(device_id: usize) -> Option<(i32, i32)> {
    // In real implementation, query device properties
    Some((8, 6)) // Placeholder: RTX 3090 level
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_device_props() {
        let props = DeviceProps {
            device_id: 0,
            name: "Test GPU".to_string(),
            compute_capability: (8, 6),
            total_memory: 24_000_000_000,
            free_memory: 20_000_000_000,
        };
        
        assert!(props.supports_tensor_cores());
        assert!(props.supports_async_copy());
    }
}
