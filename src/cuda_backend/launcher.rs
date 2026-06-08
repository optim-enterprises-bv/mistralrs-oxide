//! Kernel launcher with PTX compilation support

use std::collections::HashMap;
use std::sync::Arc;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig};
use cuda_host::{CudaModule, CudaFunction};
use crate::core::{DType, Shape};
use crate::error::{OxideError, OxideResult, bail};
use crate::cuda_backend::errors::{check_cuda, CudaResult};

/// Compiled PTX module cache
pub struct PtxModuleCache {
    modules: HashMap<String, Arc<CudaModule>>,
}

impl PtxModuleCache {
    pub fn new() -> Self {
        Self {
            modules: HashMap::new(),
        }
    }

    /// Load PTX from file
    pub fn load_ptx(
        &mut self,
        context: &Arc<CudaContext>,
        path: &str,
    ) -> OxideResult<Arc<CudaModule>> {
        if let Some(module) = self.modules.get(path) {
            return Ok(module.clone());
        }

        // Load PTX code from file
        let ptx_code = std::fs::read_to_string(path)
            .map_err(|e| OxideError::IoError(e))?;

        let module = CudaModule::load_from_ptx(context, &ptx_code)
            .map_err(|e| OxideError::CudaError(format!("Failed to load PTX: {}", e)))?;
        
        let module = Arc::new(module);
        self.modules.insert(path.to_string(), module.clone());
        
        Ok(module)
    }

    /// Get or load module
    pub fn get(&mut self, path: &str) -> Option<Arc<CudaModule>> {
        self.modules.get(path).cloned()
    }

    /// Clear cache
    pub fn clear(&mut self) {
        self.modules.clear();
    }
}

/// Kernel launcher
pub struct KernelLauncher {
    context: Arc<CudaContext>,
    module_cache: PtxModuleCache,
}

impl KernelLauncher {
    pub fn new(context: &Arc<CudaContext>) -> Self {
        Self {
            context: context.clone(),
            module_cache: PtxModuleCache::new(),
        }
    }

    /// Launch 1D kernel
    pub fn launch_1d(
        &mut self,
        stream: &CudaStream,
        function: &CudaFunction,
        num_elements: usize,
        block_size: usize,
        args: &[&dyn std::any::Any],
    ) -> OxideResult<()> {
        let grid_size = (num_elements + block_size - 1) / block_size;
        
        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        function.launch(stream, config, args)
            .map_err(|e| OxideError::CudaError(format!("Kernel launch failed: {}", e)))?;
        
        Ok(())
    }

    /// Launch 2D kernel
    pub fn launch_2d(
        &mut self,
        stream: &CudaStream,
        function: &CudaFunction,
        dim_x: usize,
        dim_y: usize,
        block_x: usize,
        block_y: usize,
        args: &[&dyn std::any::Any],
    ) -> OxideResult<()> {
        let grid_x = (dim_x + block_x - 1) / block_x;
        let grid_y = (dim_y + block_y - 1) / block_y;
        
        let config = LaunchConfig {
            grid_dim: (grid_x as u32, grid_y as u32, 1),
            block_dim: (block_x as u32, block_y as u32, 1),
            shared_mem_bytes: 0,
        };

        function.launch(stream, config, args)
            .map_err(|e| OxideError::CudaError(format!("Kernel launch failed: {}", e)))?;
        
        Ok(())
    }

    /// Launch with shared memory
    pub fn launch_with_shared_mem(
        &mut self,
        stream: &CudaStream,
        function: &CudaFunction,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        shared_mem_bytes: u32,
        args: &[&dyn std::any::Any],
    ) -> OxideResult<()> {
        let config = LaunchConfig {
            grid_dim,
            block_dim,
            shared_mem_bytes,
        };

        function.launch(stream, config, args)
            .map_err(|e| OxideError::CudaError(format!("Kernel launch failed: {}", e)))?;
        
        Ok(())
    }

    /// Load and launch kernel from PTX
    pub fn launch_from_ptx_1d(
        &mut self,
        stream: &CudaStream,
        ptx_path: &str,
        kernel_name: &str,
        num_elements: usize,
        block_size: usize,
        args: &[&dyn std::any::Any],
    ) -> OxideResult<()> {
        let module = self.module_cache.load_ptx(&self.context, ptx_path)?;
        let function = module.get_function(kernel_name)
            .ok_or_else(|| OxideError::CudaError(format!("Kernel {} not found", kernel_name)))?;
        
        self.launch_1d(stream, &function, num_elements, block_size, args)
    }
}

/// PTX compiler wrapper
pub struct PtxCompiler;

impl PtxCompiler {
    /// Compile Rust kernel code to PTX using cuda-oxide
    pub fn compile_kernel(
        source: &str,
        kernel_name: &str,
    ) -> OxideResult<String> {
        // This would invoke cuda-oxide's rustc backend
        // For now, return placeholder
        Ok(format!(
            "// PTX for kernel: {}\n.version 8.0\n.target sm_86\n.entry {}",
            kernel_name, kernel_name
        ))
    }

    /// Compile and save PTX to file
    pub fn compile_and_save(
        source: &str,
        kernel_name: &str,
        output_path: &str,
    ) -> OxideResult<()> {
        let ptx = Self::compile_kernel(source, kernel_name)?;
        std::fs::write(output_path, ptx)
            .map_err(|e| OxideError::IoError(e))?;
        Ok(())
    }
}

/// Kernel build system
pub struct KernelBuilder {
    kernels: Vec<KernelSource>,
}

#[derive(Clone)]
pub struct KernelSource {
    pub name: String,
    pub source: String,
    pub block_size: usize,
}

impl KernelBuilder {
    pub fn new() -> Self {
        Self {
            kernels: Vec::new(),
        }
    }

    /// Add kernel to build
    pub fn add_kernel(
        &mut self,
        name: &str,
        source: &str,
        block_size: usize,
    ) {
        self.kernels.push(KernelSource {
            name: name.to_string(),
            source: source.to_string(),
            block_size,
        });
    }

    /// Build all kernels
    pub fn build_all(&self, output_dir: &str) -> OxideResult<Vec<String>> {
        let mut ptx_paths = Vec::new();
        
        for kernel in &self.kernels {
            let ptx_path = format!("{}/{}.ptx", output_dir, kernel.name);
            PtxCompiler::compile_and_save(&kernel.source, &kernel.name, &ptx_path)?;
            ptx_paths.push(ptx_path);
        }
        
        Ok(ptx_paths)
    }

    /// Get kernel by name
    pub fn get_kernel(&self, name: &str) -> Option<&KernelSource> {
        self.kernels.iter().find(|k| k.name == name)
    }
}

/// Launch configuration builder
pub struct LaunchConfigBuilder {
    grid_dim: (u32, u32, u32),
    block_dim: (u32, u32, u32),
    shared_mem: u32,
}

impl LaunchConfigBuilder {
    pub fn new() -> Self {
        Self {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem: 0,
        }
    }

    /// Set 1D grid
    pub fn grid_1d(mut self, size: usize) -> Self {
        self.grid_dim = (size as u32, 1, 1);
        self
    }

    /// Set 2D grid
    pub fn grid_2d(mut self, x: usize, y: usize) -> Self {
        self.grid_dim = (x as u32, y as u32, 1);
        self
    }

    /// Set 1D block
    pub fn block_1d(mut self, size: usize) -> Self {
        self.block_dim = (size as u32, 1, 1);
        self
    }

    /// Set 2D block
    pub fn block_2d(mut self, x: usize, y: usize) -> Self {
        self.block_dim = (x as u32, y as u32, 1);
        self
    }

    /// Set shared memory
    pub fn shared_mem(mut self, bytes: u32) -> Self {
        self.shared_mem = bytes;
        self
    }

    /// Build configuration
    pub fn build(self) -> LaunchConfig {
        LaunchConfig {
            grid_dim: self.grid_dim,
            block_dim: self.block_dim,
            shared_mem_bytes: self.shared_mem,
        }
    }
}

/// Occupancy calculator
pub struct OccupancyCalculator;

impl OccupancyCalculator {
    /// Calculate optimal block size
    pub fn optimal_block_size(
        _function: &CudaFunction,
        _shared_mem_per_thread: usize,
    ) -> usize {
        // In real implementation, query device for optimal size
        // For now, return 256 which works well on most GPUs
        256
    }

    /// Calculate grid size for full occupancy
    pub fn grid_size_for_occupancy(
        block_size: usize,
        _num_sms: usize,
    ) -> usize {
        // Aim for at least 2 blocks per SM
        block_size * 2
    }
}
