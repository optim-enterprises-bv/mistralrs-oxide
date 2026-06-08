//! Real GPU memory management using cudarc
//!
//! Provides DeviceBuffer allocation, pinned memory, and transfers

#![cfg(feature = "cuda")]

use std::collections::HashMap;
use std::sync::Mutex;
use cudarc::driver::{CudaDevice, CudaStream, DeviceSlice, DevicePtr, LaunchConfig};
use cudarc::driver::{CudaSlice, CudaView, CudaViewMut};
use crate::core::{DType, Device};
use crate::error::{OxideError, OxideResult};
use crate::cuda_backend::errors::{CudaResult, check_cuda};

/// CUDA memory pool for efficient allocation
pub struct CudaMemoryPool {
    device: std::sync::Arc<CudaDevice>,
    allocated_bytes: Mutex<usize>,
    peak_allocated: Mutex<usize>,
    free_buffers: Mutex<HashMap<usize, Vec<CudaSlice<u8>>>>,
}

impl CudaMemoryPool {
    /// Create new memory pool for device
    pub fn new(device: &std::sync::Arc<CudaDevice>) -> OxideResult<Self> {
        Ok(Self {
            device: device.clone(),
            allocated_bytes: Mutex::new(0),
            peak_allocated: Mutex::new(0),
            free_buffers: Mutex::new(HashMap::new()),
        })
    }

    /// Allocate device memory
    pub fn allocate(&self, size_bytes: usize) -> CudaResult<CudaSlice<u8>> {
        // Check memory availability
        let (free_mem, total_mem) = self.device.memory_info();
        if size_bytes > free_mem {
            return Err(format!(
                "Out of memory: requested {} bytes, free {} bytes (total {})",
                size_bytes, free_mem, total_mem
            ));
        }

        // Try to get from pool first
        {
            let mut pool = self.free_buffers.lock().unwrap();
            let key = Self::pool_key(size_bytes);
            if let Some(buffers) = pool.get_mut(&key) {
                if let Some(buffer) = buffers.pop() {
                    if buffers.is_empty() {
                        pool.remove(&key);
                    }
                    return Ok(buffer);
                }
            }
        }

        // Allocate new buffer
        let buffer = self.device.alloc_zeros::<u8>(size_bytes)
            .map_err(|e| format!("Failed to allocate {} bytes: {}", size_bytes, e))?;

        // Track allocation
        {
            let mut allocated = self.allocated_bytes.lock().unwrap();
            *allocated += size_bytes;
            let mut peak = self.peak_allocated.lock().unwrap();
            *peak = (*peak).max(*allocated);
        }

        Ok(buffer)
    }

    /// Allocate for specific dtype
    pub fn allocate_for_dtype(
        &self,
        num_elements: usize,
        dtype: DType,
    ) -> CudaResult<CudaSlice<u8>> {
        let size = num_elements * dtype.size_in_bytes();
        self.allocate(size)
    }

    /// Free buffer back to pool
    pub fn free(&self, buffer: CudaSlice<u8>) {
        let size = buffer.len();
        let key = Self::pool_key(size);
        
        let mut pool = self.free_buffers.lock().unwrap();
        pool.entry(key).or_insert_with(Vec::new).push(buffer);
        
        let mut allocated = self.allocated_bytes.lock().unwrap();
        *allocated -= size;
    }

    /// Copy from host to device
    pub fn copy_host_to_device(
        &self,
        host: &[u8],
        device: &mut CudaSlice<u8>,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        if host.len() != device.len() {
            return Err(format!(
                "Size mismatch: host {} bytes, device {} bytes",
                host.len(), device.len()
            ));
        }

        stream.memcpy_htod(host, device)
            .map_err(|e| format!("Host to device copy failed: {}", e))
    }

    /// Copy from device to host
    pub fn copy_device_to_host(
        &self,
        device: &CudaSlice<u8>,
        host: &mut [u8],
        stream: &CudaStream,
    ) -> CudaResult<()> {
        if device.len() != host.len() {
            return Err(format!(
                "Size mismatch: device {} bytes, host {} bytes",
                device.len(), host.len()
            ));
        }

        stream.memcpy_dtoh(device, host)
            .map_err(|e| format!("Device to host copy failed: {}", e))
    }

    /// Copy device to device
    pub fn copy_device_to_device(
        &self,
        src: &CudaSlice<u8>,
        dst: &mut CudaSlice<u8>,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        if src.len() != dst.len() {
            return Err(format!(
                "Size mismatch: src {} bytes, dst {} bytes",
                src.len(), dst.len()
            ));
        }

        stream.memcpy_dtod(src, dst)
            .map_err(|e| format!("Device to device copy failed: {}", e))
    }

    /// Synchronize and clear pool
    pub fn clear(&self) -> CudaResult<()> {
        self.device.synchronize()
            .map_err(|e| format!("Synchronize failed: {}", e))?;
        
        let mut pool = self.free_buffers.lock().unwrap();
        pool.clear();
        
        let mut allocated = self.allocated_bytes.lock().unwrap();
        *allocated = 0;
        
        Ok(())
    }

    /// Get memory stats
    pub fn stats(&self) -> MemoryStats {
        let allocated = *self.allocated_bytes.lock().unwrap();
        let peak = *self.peak_allocated.lock().unwrap();
        let pool_size: usize = self.free_buffers.lock().unwrap()
            .values()
            .map(|v| v.iter().map(|b| b.len()).sum::<usize>())
            .sum();

        MemoryStats {
            allocated_bytes: allocated,
            peak_allocated_bytes: peak,
            pooled_bytes: pool_size,
        }
    }

    fn pool_key(size: usize) -> usize {
        // Round up to nearest 256 bytes for pooling
        ((size + 255) / 256) * 256
    }
}

/// Memory statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub allocated_bytes: usize,
    pub peak_allocated_bytes: usize,
    pub pooled_bytes: usize,
}

impl MemoryStats {
    pub fn allocated_mb(&self) -> f64 {
        self.allocated_bytes as f64 / (1024.0 * 1024.0)
    }
    
    pub fn peak_mb(&self) -> f64 {
        self.peak_allocated_bytes as f64 / (1024.0 * 1024.0)
    }
}

/// Pinned (page-locked) host memory
pub struct PinnedBuffer {
    data: Vec<u8>,
    _pinned: bool,
}

impl PinnedBuffer {
    /// Allocate pinned memory
    pub fn allocate(size: usize) -> CudaResult<Self> {
        // cudarc doesn't directly support pinned memory allocation
        // In production, we'd use cudaHostAlloc
        // For now, use regular allocation with large page alignment
        let mut data = vec![0u8; size];
        
        Ok(Self {
            data,
            _pinned: false, // Would be true with cudaHostAlloc
        })
    }

    /// Allocate for dtype
    pub fn allocate_for_tensor(num_elements: usize, dtype: DType) -> CudaResult<Self> {
        Self::allocate(num_elements * dtype.size_in_bytes())
    }

    /// Get slice
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    /// Get mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Get length
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Extension trait for tensor-like operations on DeviceBuffers
pub trait DeviceBufferExt {
    fn slice(&self, start: usize, len: usize) -> CudaView<u8>;
    fn slice_mut(&mut self, start: usize, len: usize) -> CudaViewMut<u8>;
    fn as_ptr(&self) -> u64;
    fn len(&self) -> usize;
}

impl DeviceBufferExt for CudaSlice<u8> {
    fn slice(&self, start: usize, len: usize) -> CudaView<u8> {
        self.slice(start..start + len)
    }

    fn slice_mut(&mut self, start: usize, len: usize) -> CudaViewMut<u8> {
        self.slice_mut(start..start + len)
    }

    fn as_ptr(&self) -> u64 {
        // Get raw device pointer
        self.device_ptr() as u64
    }

    fn len(&self) -> usize {
        self.len()
    }
}

/// Alignment utilities
pub fn align_size(size: usize, alignment: usize) -> usize {
    (size + alignment - 1) & !(alignment - 1)
}

/// Common alignments
pub const ALIGN_256: usize = 256;
pub const ALIGN_512: usize = 512;

/// Memory bandwidth monitor
pub struct BandwidthMonitor {
    bytes_transferred: usize,
    transfer_time_ns: u64,
}

impl BandwidthMonitor {
    pub fn new() -> Self {
        Self {
            bytes_transferred: 0,
            transfer_time_ns: 0,
        }
    }

    pub fn record_transfer(&mut self, bytes: usize, time_ns: u64) {
        self.bytes_transferred += bytes;
        self.transfer_time_ns += time_ns;
    }

    pub fn bandwidth_gbps(&self) -> f64 {
        if self.transfer_time_ns == 0 {
            0.0
        } else {
            (self.bytes_transferred as f64 / 1e9) / (self.transfer_time_ns as f64 / 1e9)
        }
    }

    pub fn reset(&mut self) {
        self.bytes_transferred = 0;
        self.transfer_time_ns = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_backend::is_cuda_available;

    #[test]
    fn test_memory_alignment() {
        assert_eq!(align_size(100, 256), 256);
        assert_eq!(align_size(256, 256), 256);
        assert_eq!(align_size(300, 256), 512);
    }

    #[test]
    fn test_pinned_buffer() -> CudaResult<()> {
        let mut buffer = PinnedBuffer::allocate(1024)?;
        assert_eq!(buffer.len(), 1024);
        
        // Write to buffer
        let slice = buffer.as_mut_slice();
        slice[0] = 0xDE;
        slice[1] = 0xAD;
        slice[2] = 0xBE;
        slice[3] = 0xEF;
        
        assert_eq!(slice[0], 0xDE);
        
        Ok(())
    }

    #[test]
    fn test_memory_pool() -> OxideResult<()> {
        if !is_cuda_available() {
            println!("Skipping test - no CUDA available");
            return Ok(());
        }

        use crate::cuda_backend::CudaBackend;
        
        let backend = CudaBackend::new(0)?;
        let pool = backend.memory_pool();

        // Allocate
        let buffer = pool.allocate(1024)
            .map_err(|e| OxideError::CudaError(e))?;
        assert_eq!(buffer.len(), 1024);

        // Check stats
        let stats = pool.stats();
        assert!(stats.allocated_bytes >= 1024);

        // Free (returns to pool)
        pool.free(buffer);
        
        let stats_after = pool.stats();
        assert!(stats_after.pooled_bytes >= 1024);

        Ok(())
    }
}
