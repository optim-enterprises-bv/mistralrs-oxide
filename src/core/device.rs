use std::sync::Arc;
use crate::core::{DType, Layout};
use crate::error::{OxideError, OxideResult};

#[derive(Debug, Clone)]
pub enum Device {
    Cpu,
    Cuda(CudaDevice),
}

#[derive(Debug, Clone)]
pub struct CudaDevice {
    device_id: usize,
}

impl CudaDevice {
    pub fn new(device_id: usize) -> OxideResult<Self> {
        Ok(Self { device_id })
    }

    pub fn device_id(&self) -> usize {
        self.device_id
    }
}

impl Device {
    pub fn new_cuda(device_id: usize) -> OxideResult<Self> {
        Ok(Device::Cuda(CudaDevice::new(device_id)?))
    }

    pub fn is_cuda(&self) -> bool {
        matches!(self, Device::Cuda(_))
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::Cpu)
    }

    pub fn same_device(&self, other: &Device) -> bool {
        match (self, other) {
            (Device::Cpu, Device::Cpu) => true,
            (Device::Cuda(a), Device::Cuda(b)) => a.device_id == b.device_id,
            _ => false,
        }
    }
}

impl Default for Device {
    fn default() -> Self {
        Device::Cpu
    }
}
