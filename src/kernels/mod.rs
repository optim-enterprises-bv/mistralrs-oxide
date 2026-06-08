//! Kernel module with CUDA support using cuda-oxide
//! 
//! This module provides GPU-accelerated kernels for LLM operations.
//! When cuda-oxide is properly configured, kernels compile Rust to PTX
//! and execute on NVIDIA GPUs.

pub mod elementwise;
pub mod matmul;
pub mod attention;
pub mod normalization;
pub mod execution;
pub mod integration;

pub use execution::*;
pub use integration::*;

/// Re-export kernel types
pub mod types {
    pub use super::execution::{
        KernelConfig,
        KernelExecutor,
        GpuMemoryManager,
        GpuStream,
        MatmulKernelType,
        SoftmaxKernelType,
        TileConfig,
        AutoKernelSelector,
        KernelProfiler,
        KernelProfile,
    };
}

/// GPU operation dispatch utilities
pub mod dispatch {
    pub use super::integration::{
        GpuExecutable,
        GpuOpDispatcher,
        GpuFallback,
        matmul_gpu_dispatch,
        elementwise_gpu_dispatch,
        rms_norm_gpu_dispatch,
        attention_gpu_dispatch,
        rope_gpu_dispatch,
        define_gpu_op,
    };
}

/// Initialize CUDA context for the specified device
pub fn init_cuda(device_id: usize) -> crate::error::OxideResult<> {
    // In real implementation, this would:
    // 1. Check if CUDA is available
    // 2. Initialize the CUDA context
    // 3. Set up the device
    
    // For now, this is a placeholder
    Ok(())
}

/// Check if CUDA is available on this system
pub fn cuda_available() -> bool {
    // In real implementation, check for CUDA devices
    false
}

/// Get CUDA device count
pub fn cuda_device_count() -> usize {
    // In real implementation, query CUDA for device count
    0
}

/// Get CUDA device properties
pub fn cuda_device_props(device_id: usize) -> Option<CudaDeviceProps> {
    None
}

/// CUDA device properties
#[derive(Debug, Clone)]
pub struct CudaDeviceProps {
    pub name: String,
    pub compute_capability_major: i32,
    pub compute_capability_minor: i32,
    pub total_memory: usize,
    pub max_threads_per_block: i32,
    pub warp_size: i32,
}

/// Compile-time kernel metadata
pub struct KernelMetadata {
    pub name: &'static str,
    pub num_regs: usize,
    pub shared_mem_bytes: usize,
    pub const_mem_bytes: usize,
}

/// Get metadata for all compiled kernels
pub fn get_kernel_metadata() -> Vec<KernelMetadata> {
    vec![
        KernelMetadata {
            name: "vecadd_f32",
            num_regs: 16,
            shared_mem_bytes: 0,
            const_mem_bytes: 0,
        },
        KernelMetadata {
            name: "matmul_f32",
            num_regs: 32,
            shared_mem_bytes: 2048,
            const_mem_bytes: 0,
        },
        KernelMetadata {
            name: "attention_forward_f32",
            num_regs: 64,
            shared_mem_bytes: 4096,
            const_mem_bytes: 256,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cuda_availability() {
        // This will return false without actual CUDA hardware
        let available = cuda_available();
        println!("CUDA available: {}", available);
    }
    
    #[test]
    fn test_kernel_metadata() {
        let metadata = get_kernel_metadata();
        assert!(!metadata.is_empty());
        for meta in metadata {
            println!("Kernel: {} (regs: {}, shared: {})", 
                meta.name, meta.num_regs, meta.shared_mem_bytes);
        }
    }
}
