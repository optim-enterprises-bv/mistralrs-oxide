//! CUDA context management using cudarc

#![cfg(feature = "cuda")]

use std::sync::Arc;
use cudarc::driver::{CudaDevice, CudaContext as CudaCtx};
use crate::error::{OxideError, OxideResult};

/// Manages CUDA device contexts
pub struct CudaContextManager {
    contexts: Vec<Arc<CudaCtx>>,
    current: usize,
}

impl CudaContextManager {
    /// Create contexts for all available devices
    pub fn new() -> OxideResult<Self> {
        let device_count = cudarc::driver::device_count()
            .map_err(|e| OxideError::CudaError(format!("Failed to get device count: {}", e)))?;
        
        let mut contexts = Vec::with_capacity(device_count);
        
        for i in 0..device_count {
            let device = CudaDevice::new(i)
                .map_err(|e| OxideError::CudaError(format!("Failed to create device {}: {}", i, e)))?;
            contexts.push(device.default_context());
        }
        
        Ok(Self {
            contexts,
            current: 0,
        })
    }
    
    /// Get context for device
    pub fn get(&self, device_id: usize) -> Option<&Arc<CudaCtx>> {
        self.contexts.get(device_id)
    }
    
    /// Set current context
    pub fn set_current(&mut self, device_id: usize) -> OxideResult<()> {
        if device_id >= self.contexts.len() {
            return Err(OxideError::InvalidArgument(
                format!("Device {} out of range", device_id)
            ));
        }
        self.current = device_id;
        Ok(())
    }
    
    /// Get current context
    pub fn current(&self) -> &Arc<CudaCtx> {
        &self.contexts[self.current]
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
    pub shared_memory_per_block: usize,
    pub max_threads_per_block: usize,
    pub max_block_dim: (usize, usize, usize),
    pub max_grid_dim: (usize, usize, usize),
    pub warp_size: usize,
    pub memory_clock_rate: i32,
    pub memory_bus_width: i32,
    pub l2_cache_size: i32,
}

impl DeviceProperties {
    /// Get properties from device
    pub fn from_device(device: &Arc<CudaDevice>, device_id: usize) -> Self {
        // Get device attributes
        let major = device.attribute(cudarc::driver::DeviceAttribute::ComputeCapabilityMajor)
            .unwrap_or(0) as i32;
        let minor = device.attribute(cudarc::driver::DeviceAttribute::ComputeCapabilityMinor)
            .unwrap_or(0) as i32;
        
        let (free, total) = device.memory_info();
        
        DeviceProperties {
            device_id,
            name: format!("CUDA Device {}", device_id),
            major,
            minor,
            total_memory: total,
            shared_memory_per_block: 49152, // 48KB typical
            max_threads_per_block: 1024,
            max_block_dim: (1024, 1024, 64),
            max_grid_dim: (2147483647, 65535, 65535),
            warp_size: 32,
            memory_clock_rate: 8000,
            memory_bus_width: 256,
            l2_cache_size: 4 * 1024 * 1024, // 4MB typical
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
    
    /// Calculate memory bandwidth in GB/s
    pub fn memory_bandwidth_gbps(&self) -> f64 {
        let clock_hz = self.memory_clock_rate as f64 * 1000.0;
        let bus_bytes = (self.memory_bus_width / 8) as f64;
        (clock_hz * 2.0 * bus_bytes) / 1e9
    }
}

/// Context configuration
#[derive(Debug, Clone)]
pub struct ContextConfig {
    pub device_id: usize,
    pub flags: u32,
}

impl ContextConfig {
    pub fn new(device_id: usize) -> Self {
        Self {
            device_id,
            flags: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_device_properties() {
        use std::sync::Arc;
        
        // Create mock device (this won't work without real CUDA)
        // Just test the struct
        let props = DeviceProperties {
            device_id: 0,
            name: "RTX 3060".to_string(),
            major: 8,
            minor: 6,
            total_memory: 6 * 1024 * 1024 * 1024,
            shared_memory_per_block: 48 * 1024,
            max_threads_per_block: 1024,
            max_block_dim: (1024, 1024, 64),
            max_grid_dim: (2147483647, 65535, 65535),
            warp_size: 32,
            memory_clock_rate: 8000,
            memory_bus_width: 256,
            l2_cache_size: 4 * 1024 * 1024,
        };
        
        assert!(props.has_tensor_cores());
        assert!(props.supports_async_copy());
    }
}
