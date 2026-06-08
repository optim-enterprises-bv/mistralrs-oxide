//! GPU memory management with cuda-core DeviceBuffer

use std::collections::HashMap;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use crate::core::{DType, Device};
use crate::error::{OxideError, OxideResult, bail};

/// Memory pool for efficient GPU memory allocation
pub struct GpuMemoryPool {
    device_id: usize,
    allocations: HashMap<usize, Vec<DeviceBuffer<u8>>>,
    used_bytes: usize,
    peak_bytes: usize,
}

impl GpuMemoryPool {
    pub fn new(device_id: usize) -> OxideResult<Self> {
        Ok(Self {
            device_id,
            allocations: HashMap::new(),
            used_bytes: 0,
            peak_bytes: 0,
        })
    }

    /// Allocate device buffer
    pub fn allocate(
        &mut self,
        stream: &CudaStream,
        size_bytes: usize,
    ) -> OxideResult<DeviceBuffer<u8>> {
        let buffer = DeviceBuffer::zeroed(stream, size_bytes)
            .map_err(|e| OxideError::CudaError(format!("Allocation failed: {}", e)))?;
        
        self.used_bytes += size_bytes;
        self.peak_bytes = self.peak_bytes.max(self.used_bytes);
        
        Ok(buffer)
    }

    /// Allocate typed buffer
    pub fn allocate_typed<T>(
        &mut self,
        stream: &CudaStream,
        num_elements: usize,
    ) -> OxideResult<DeviceBuffer<T>> {
        let size_bytes = num_elements * std::mem::size_of::<T>();
        // Convert bytes to T (this is a simplification)
        DeviceBuffer::zeroed(stream, size_bytes)
            .map_err(|e| OxideError::CudaError(format!("Allocation failed: {}", e)))
    }

    /// Allocate for dtype
    pub fn allocate_for_tensor(
        &mut self,
        stream: &CudaStream,
        num_elements: usize,
        dtype: DType,
    ) -> OxideResult<DeviceBuffer<u8>> {
        let size_bytes = num_elements * dtype.size_in_bytes();
        self.allocate(stream, size_bytes)
    }

    /// Free buffer back to pool
    pub fn free(&mut self, buffer: DeviceBuffer<u8>) {
        // In a real pool, we'd cache this for reuse
        // For now, just drop it
        drop(buffer);
    }

    /// Get memory stats
    pub fn stats(&self) -> MemoryStats {
        MemoryStats {
            used_bytes: self.used_bytes,
            peak_bytes: self.peak_bytes,
            device_id: self.device_id,
        }
    }
}

/// Memory statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub used_bytes: usize,
    pub peak_bytes: usize,
    pub device_id: usize,
}

impl MemoryStats {
    pub fn used_mb(&self) -> f64 {
        self.used_bytes as f64 / (1024.0 * 1024.0)
    }
    
    pub fn peak_mb(&self) -> f64 {
        self.peak_bytes as f64 / (1024.0 * 1024.0)
    }
}

/// Manages device buffers for tensors
pub struct DeviceBufferManager {
    device: Device,
    buffers: HashMap<usize, DeviceBuffer<u8>>,
}

impl DeviceBufferManager {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
            buffers: HashMap::new(),
        }
    }

    /// Create buffer from host data
    pub fn from_host<T: Copy>(
        &mut self,
        stream: &CudaStream,
        data: &[T],
    ) -> OxideResult<DeviceBuffer<u8>> {
        let bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<T>(),
            ).to_vec()
        };
        
        DeviceBuffer::from_host(stream, &bytes)
            .map_err(|e| OxideError::CudaError(format!("Host to device copy failed: {}", e)))
    }

    /// Copy device buffer to host
    pub fn to_host<T: Copy>(
        &self,
        stream: &CudaStream,
        buffer: &DeviceBuffer<u8>,
        num_elements: usize,
    ) -> OxideResult<Vec<T>> {
        let bytes = buffer.to_host_vec(stream)
            .map_err(|e| OxideError::CudaError(format!("Device to host copy failed: {}", e)))?;
        
        if bytes.len() != num_elements * std::mem::size_of::<T>() {
            bail!("Size mismatch in host copy");
        }
        
        let result: Vec<T> = unsafe {
            let ptr = bytes.as_ptr() as *const T;
            std::slice::from_raw_parts(ptr, num_elements).to_vec()
        };
        
        Ok(result)
    }

    /// Allocate aligned buffer
    pub fn allocate_aligned(
        &mut self,
        stream: &CudaStream,
        size_bytes: usize,
        alignment: usize,
    ) -> OxideResult<DeviceBuffer<u8>> {
        // CUDA allocations are typically 256-byte aligned
        let aligned_size = (size_bytes + alignment - 1) & !(alignment - 1);
        DeviceBuffer::zeroed(stream, aligned_size)
            .map_err(|e| OxideError::CudaError(format!("Aligned allocation failed: {}", e)))
    }

    /// Clear all managed buffers
    pub fn clear(&mut self) {
        self.buffers.clear();
    }
}

/// Pinned (page-locked) host memory for faster transfers
pub struct PinnedMemory {
    _marker: std::marker::PhantomData<u8>,
}

impl PinnedMemory {
    /// Allocate pinned host memory
    pub fn allocate(size_bytes: usize) -> OxideResult<Vec<u8>> {
        // In real implementation, use cudaHostAlloc
        // For now, return regular vec
        Ok(vec![0u8; size_bytes])
    }

    /// Allocate for tensor
    pub fn allocate_for_tensor(num_elements: usize, dtype: DType) -> OxideResult<Vec<u8>> {
        Self::allocate(num_elements * dtype.size_in_bytes())
    }
}

/// Memory bandwidth tracking
pub struct BandwidthMonitor {
    bytes_transferred: usize,
    transfer_time_ms: f64,
}

impl BandwidthMonitor {
    pub fn new() -> Self {
        Self {
            bytes_transferred: 0,
            transfer_time_ms: 0.0,
        }
    }

    pub fn record_transfer(&mut self, bytes: usize, time_ms: f64) {
        self.bytes_transferred += bytes;
        self.transfer_time_ms += time_ms;
    }

    pub fn bandwidth_gbps(&self) -> f64 {
        if self.transfer_time_ms == 0.0 {
            0.0
        } else {
            (self.bytes_transferred as f64 / 1e9) / (self.transfer_time_ms / 1000.0)
        }
    }

    pub fn reset(&mut self) {
        self.bytes_transferred = 0;
        self.transfer_time_ms = 0.0;
    }
}
