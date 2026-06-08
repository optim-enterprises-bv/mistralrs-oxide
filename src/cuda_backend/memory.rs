//! Working CUDA memory management

#![cfg(feature = "cuda")]

use cudarc::driver::CudaDevice;

/// Simple memory pool
pub struct CudaMemoryPool {
    device: std::sync::Arc<CudaDevice>,
}

impl CudaMemoryPool {
    pub fn new(device: &std::sync::Arc<CudaDevice>) -> Result<Self, String> {
        Ok(Self {
            device: device.clone(),
        })
    }

    /// Allocate f32 buffer
    pub fn allocate_f32(
        &self, 
        len: usize
    ) -> Result<cudarc::driver::CudaSlice<f32>, String> {
        self.device.alloc_zeros::<f32>(len)
            .map_err(|e| format!("Failed to allocate: {}", e))
    }
}
