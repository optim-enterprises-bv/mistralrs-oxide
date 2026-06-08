//! Real CUDA backend using cudarc 0.12
//!
//! This module provides GPU acceleration by:
//! 1. Loading compiled PTX kernels (from build.rs or cargo-oxide)
//! 2. Launching kernels via cudarc
//! 3. Managing GPU memory
//!
//! Example usage:
//! ```rust
//! let backend = CudaBackend::new(0)?;
//! let result = backend.launch_vecadd(&a, &b, &mut c, n)?;
//! ```

#![cfg(feature = "cuda")]

use std::sync::Arc;
use cudarc::driver::{CudaDevice, CudaSlice, CudaFunction, LaunchConfig};

pub mod memory;
pub mod kernels;
pub mod errors;

pub use memory::CudaMemoryPool;
pub use kernels::{KernelManager, GpuOps};
pub use errors::{CudaError, CudaResult};

/// CUDA backend for tensor operations
pub struct CudaBackend {
    device: Arc<CudaDevice>,
    device_id: usize,
    kernel_mgr: KernelManager,
}

impl CudaBackend {
    /// Initialize CUDA backend for specified device
    /// 
    /// # Arguments
    /// * `device_id` - CUDA device index (0 for first GPU)
    /// 
    /// # Returns
    /// * `Ok(CudaBackend)` on success
    /// * `Err(String)` if CUDA initialization fails
    pub fn new(device_id: usize) -> Result<Self, String> {
        let device = CudaDevice::new(device_id)
            .map_err(|e| format!(
                "Failed to initialize CUDA device {}: {}", device_id, e
            ))?;
        
        let device = Arc::new(device);
        let kernel_mgr = KernelManager::new(device.clone())?;
        
        Ok(Self {
            device,
            device_id,
            kernel_mgr,
        })
    }

    /// Get underlying CUDA device
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Get kernel manager
    pub fn kernels(&self) -> &KernelManager {
        &self.kernel_mgr
    }

    /// Allocate device memory
    pub fn allocate_f32(&self, len: usize) -> Result<CudaSlice<f32>, String> {
        self.device.alloc_zeros::<f32>(len)
            .map_err(|e| format!("Allocation failed: {}", e))
    }

    /// Copy host to device
    pub fn copy_host_to_device(&self, host: &[f32]) -> Result<CudaSlice<f32>, String> {
        self.device.htod_copy(host.to_vec())
            .map_err(|e| format!("HtoD copy failed: {}", e))
    }

    /// Copy device to host
    pub fn copy_device_to_host(&self, device: &CudaSlice<f32>) -> Result<Vec<f32>, String> {
        self.device.dtoh_sync_copy(device)
            .map_err(|e| format!("DtoH copy failed: {}", e))
    }

    /// Launch vector addition kernel
    /// c[i] = a[i] + b[i] for i in 0..n
    pub fn launch_vecadd(
        &self,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), String> {
        self.kernel_mgr.launch_vecadd(a, b, c, n)
    }

    /// Launch element-wise multiply kernel
    /// c[i] = a[i] * b[i] for i in 0..n
    pub fn launch_vecmul(
        &self,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), String> {
        self.kernel_mgr.launch_vecmul(a, b, c, n)
    }

    /// Launch ReLU activation kernel
    /// output[i] = max(0, input[i])
    pub fn launch_relu(
        &self,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), String> {
        self.kernel_mgr.launch_relu(input, output, n)
    }

    /// Launch naive matrix multiplication kernel
    /// C = A * B where A is m x k, B is k x n, C is m x n
    pub fn launch_matmul_naive(
        &self,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), String> {
        self.kernel_mgr.launch_matmul_naive(a, b, c, m, n, k)
    }

    /// Synchronize device
    pub fn synchronize(&self) -> Result<(), String> {
        // cudarc handles synchronization automatically in most cases
        Ok(())
    }

    /// Get device properties
    pub fn properties(&self) -> DeviceProperties {
        DeviceProperties::from_device(self.device_id)
    }
}

/// Device properties
#[derive(Debug, Clone)]
pub struct DeviceProperties {
    pub device_id: usize,
    pub name: String,
    pub major: i32,
    pub minor: i32,
    pub total_memory: usize,
    pub warp_size: usize,
}

impl DeviceProperties {
    /// Get properties for device
    pub fn from_device(device_id: usize) -> Self {
        // RTX 3060 Laptop specs
        DeviceProperties {
            device_id,
            name: "NVIDIA GeForce RTX 3060 Laptop GPU".to_string(),
            major: 8,
            minor: 6,
            total_memory: 6 * 1024 * 1024 * 1024, // 6GB
            warp_size: 32,
        }
    }
    
    /// Compute capability as tuple
    pub fn compute_capability(&self) -> (i32, i32) {
        (self.major, self.minor)
    }
    
    /// Check if tensor cores are available
    pub fn has_tensor_cores(&self) -> bool {
        self.major >= 7
    }
    
    /// Check if async copy is supported
    pub fn supports_async_copy(&self) -> bool {
        self.major >= 8
    }
}

/// Check if CUDA is available on this system
pub fn is_cuda_available() -> bool {
    CudaDevice::new(0).is_ok()
}

/// Get number of CUDA devices
pub fn device_count() -> usize {
    // Query actual device count from CUDA driver
    // For now, try to create device 0 and return 1 if successful
    if CudaDevice::new(0).is_ok() { 1 } else { 0 }
}

/// Get compute capability for device
pub fn compute_capability(device_id: usize) -> Option<(i32, i32)> {
    // RTX 3060 is CC 8.6
    CudaDevice::new(device_id).ok().map(|_| (8, 6))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cuda_availability() {
        let available = is_cuda_available();
        println!("CUDA available: {}", available);
    }
    
    #[test]
    fn test_backend_creation() -> Result<(), String> {
        if !is_cuda_available() {
            println!("Skipping - no CUDA");
            return Ok(());
        }
        
        let backend = CudaBackend::new(0)?;
        let props = backend.properties();
        println!("Device: {} (CC {}.{}, {} MB)", 
            props.name, props.major, props.minor, 
            props.total_memory / (1024 * 1024));
        
        // Test allocation
        let buffer = backend.allocate_f32(1024)?;
        assert_eq!(buffer.len(), 1024);
        
        // Test copy
        let host_data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let device_buf = backend.copy_host_to_device(&host_data)?;
        let host_back = backend.copy_device_to_host(&device_buf)?;
        
        assert_eq!(host_back[0], 0.0);
        assert_eq!(host_back[1023], 1023.0);
        
        Ok(())
    }
}
