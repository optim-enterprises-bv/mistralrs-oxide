//! Kernel manager - loads PTX and launches kernels

#![cfg(feature = "cuda")]

use std::collections::HashMap;
use std::sync::Arc;
use cudarc::driver::{CudaDevice, CudaSlice, CudaFunction, LaunchConfig};
use cudarc::nvrtc::Ptx;

/// Manages compiled kernels
pub struct KernelManager {
    device: Arc<CudaDevice>,
    functions: HashMap<String, CudaFunction>,
}

impl KernelManager {
    /// Create new kernel manager and load built-in kernels
    pub fn new(device: Arc<CudaDevice>) -> Result<Self, String> {
        let mut mgr = Self {
            device,
            functions: HashMap::new(),
        };
        
        // Load built-in kernels from PTX
        mgr.load_builtin_kernels()?;
        
        Ok(mgr)
    }

    /// Load built-in kernels
    fn load_builtin_kernels(&mut self) -> Result<(), String> {
        // PTX code generated at build time
        let ptx = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));
        
        let ptx = Ptx::from_src(ptx)
            .map_err(|e| format!("Failed to parse PTX: {}", e))?;
        
        // Load all kernels
        let kernel_names = vec![
            "vecadd_f32",
            "vecmul_f32",
            "relu_f32",
            "matmul_naive_f32",
        ];
        
        self.device.load_ptx(ptx, "kernels", &kernel_names)
            .map_err(|e| format!("Failed to load PTX: {}", e))?;
        
        // Cache function handles
        for name in kernel_names {
            if let Some(func) = self.device.get_func("kernels", name) {
                self.functions.insert(name.to_string(), func);
            }
        }
        
        Ok(())
    }

    /// Get kernel function by name
    fn get_kernel(&self, name: &str) -> Option<&CudaFunction> {
        self.functions.get(name)
    }

    /// Launch vector addition: c = a + b
    pub fn launch_vecadd(
        &self,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), String> {
        let func = self.get_kernel("vecadd_f32")
            .ok_or("Kernel vecadd_f32 not loaded")?;
        
        let cfg = LaunchConfig::for_num_elems(n as u32);
        
        // Launch kernel with arguments
        unsafe {
            func.launch(cfg, (a, b, c, n as i32))
                .map_err(|e| format!("vecadd launch failed: {}", e))?;
        }
        
        Ok(())
    }

    /// Launch vector multiply: c = a * b
    pub fn launch_vecmul(
        &self,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), String> {
        let func = self.get_kernel("vecmul_f32")
            .ok_or("Kernel vecmul_f32 not loaded")?;
        
        let cfg = LaunchConfig::for_num_elems(n as u32);
        
        unsafe {
            func.launch(cfg, (a, b, c, n as i32))
                .map_err(|e| format!("vecmul launch failed: {}", e))?;
        }
        
        Ok(())
    }

    /// Launch ReLU: output = max(0, input)
    pub fn launch_relu(
        &self,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), String> {
        let func = self.get_kernel("relu_f32")
            .ok_or("Kernel relu_f32 not loaded")?;
        
        let cfg = LaunchConfig::for_num_elems(n as u32);
        
        unsafe {
            func.launch(cfg, (input, output, n as i32))
                .map_err(|e| format!("relu launch failed: {}", e))?;
        }
        
        Ok(())
    }

    /// Launch matrix multiplication: C = A * B
    pub fn launch_matmul_naive(
        &self,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), String> {
        let func = self.get_kernel("matmul_naive_f32")
            .ok_or("Kernel matmul_naive_f32 not loaded")?;
        
        // Use 2D grid for matrix
        let block_size = 16;
        let grid_x = ((m + block_size - 1) / block_size) as u32;
        let grid_y = ((n + block_size - 1) / block_size) as u32;
        
        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (block_size as u32, block_size as u32, 1),
            shared_mem_bytes: 0,
        };
        
        unsafe {
            func.launch(cfg, (a, b, c, m as i32, n as i32, k as i32))
                .map_err(|e| format!("matmul launch failed: {}", e))?;
        }
        
        Ok(())
    }
}

/// GPU operations using kernels
pub struct GpuOps {
    _device: Arc<CudaDevice>,
}

impl GpuOps {
    pub fn new(device: Arc<CudaDevice>) -> Result<Self, String> {
        Ok(Self { _device: device })
    }
}
