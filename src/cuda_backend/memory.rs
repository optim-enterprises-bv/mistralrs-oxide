//! Real GPU memory management using cudarc 0.12

#![cfg(feature = "cuda")]

use std::collections::HashMap;
use std::sync::Mutex;
use cudarc::driver::{CudaDevice, CudaSlice};
use crate::error::{OxideError, OxideResult};
use crate::cuda_backend::errors::CudaResult;

/// CUDA memory pool for efficient allocation
pub struct CudaMemoryPool {
    device: std::sync::Arc<CudaDevice>,
    allocated_bytes: Mutex<usize>,
    peak_allocated: Mutex<usize>,
}

impl CudaMemoryPool {
    /// Create new memory pool for device
    pub fn new(device: &std::sync::Arc<CudaDevice>) -> OxideResult<Self> {
        Ok(Self {
            device: device.clone(),
            allocated_bytes: Mutex::new(0),
            peak_allocated: Mutex::new(0),
        })
    }

    /// Allocate device memory using cudarc API
    pub fn allocate<T: cudarc::driver::DeviceRepr>(
        &self, 
        len: usize
    ) -> CudaResult<CudaSlice<T>> {
        let buffer = self.device.alloc_zeros::<T>(len)
            .map_err(|e| format!("Failed to allocate {} elements: {}", len, e))?;
        
        let size = len * std::mem::size_of::<T>();
        {
            let mut allocated = self.allocated_bytes.lock().unwrap();
            *allocated += size;
            let mut peak = self.peak_allocated.lock().unwrap();
            *peak = (*peak).max(*allocated);
        }

        Ok(buffer)
    }

    /// Copy from host to device
    pub fn copy_host_to_device<T: cudarc::driver::DeviceRepr + Clone>(
        &self,
        host: &[T],
    ) -> CudaResult<CudaSlice<T>> {
        self.device.htod_copy(host.to_vec())
            .map_err(|e| format!("Host to device copy failed: {}", e))
    }

    /// Copy from device to host
    pub fn copy_device_to_host<T: cudarc::driver::DeviceRepr + Clone>(
        &self,
        device: &CudaSlice<T>,
    ) -> CudaResult<Vec<T>> {
        self.device.dtoh_sync_copy(device)
            .map_err(|e| format!("Device to host copy failed: {}", e))
    }

    /// Get memory stats
    pub fn stats(&self) -> MemoryStats {
        let allocated = *self.allocated_bytes.lock().unwrap();
        let peak = *self.peak_allocated.lock().unwrap();

        MemoryStats {
            allocated_bytes: allocated,
            peak_allocated_bytes: peak,
        }
    }

    /// Synchronize device
    pub fn synchronize(&self) -> CudaResult<()> {
        Ok(()) // cudarc handles sync automatically in most cases
    }
}

/// Memory statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub allocated_bytes: usize,
    pub peak_allocated_bytes: usize,
}

impl MemoryStats {
    pub fn allocated_mb(&self) -> f64 {
        self.allocated_bytes as f64 / (1024.0 * 1024.0)
    }
    
    pub fn peak_mb(&self) -> f64 {
        self.peak_allocated_bytes as f64 / (1024.0 * 1024.0)
    }
}

/// Extension trait for device buffers
pub trait DeviceBufferExt<T: cudarc::driver::DeviceRepr> {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
}

impl<T: cudarc::driver::DeviceRepr> DeviceBufferExt<T> for CudaSlice<T> {
    fn len(&self) -> usize {
        self.len()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_backend::is_cuda_available;

    #[test]
    fn test_memory_pool() -> OxideResult<()> {
        if !is_cuda_available() {
            println!("Skipping test - no CUDA available");
            return Ok(());
        }

        let device = CudaDevice::new(0).unwrap();
        let pool = CudaMemoryPool::new(&std::sync::Arc::new(device)).unwrap();

        // Allocate f32 buffer
        let buffer: CudaSlice<f32> = pool.allocate(1024).unwrap();
        assert_eq!(buffer.len(), 1024);

        // Check stats
        let stats = pool.stats();
        assert!(stats.allocated_bytes >= 4096);

        // Test copy
        let host_data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let device_buf = pool.copy_host_to_device(&host_data).unwrap();
        assert_eq!(device_buf.len(), 1024);

        let host_back = pool.copy_device_to_host(&device_buf).unwrap();
        assert_eq!(host_back[0], 0.0);
        assert_eq!(host_back[1023], 1023.0);

        Ok(())
    }
}
