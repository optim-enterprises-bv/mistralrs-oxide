use crate::core::{Device, DType, Layout};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Storage {
    pub dtype: DType,
    pub layout: Layout,
    pub data: StorageData,
}

#[derive(Debug, Clone)]
pub enum StorageData {
    Cpu(CpuStorage),
    Cuda(CudaStorage),
}

#[derive(Debug, Clone)]
pub struct CpuStorage {
    pub data: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct CudaStorage {
    pub device_id: usize,
    pub ptr: usize,
    pub len: usize,
}

impl Storage {
    pub fn device(&self) -> Device {
        match &self.data {
            StorageData::Cpu(_) => Device::Cpu,
            StorageData::Cuda(cuda) => Device::Cuda(crate::core::CudaDevice::new(cuda.device_id).unwrap()),
        }
    }

    pub fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }

    pub fn to_device(&self, device: &Device) -> crate::error::OxideResult<Storage> {
        if self.device().same_device(device) {
            return Ok(self.clone());
        }

        match (&self.data, device) {
            (StorageData::Cpu(cpu), Device::Cuda(cuda)) => {
                Ok(Storage {
                    dtype: self.dtype,
                    layout: self.layout.clone(),
                    data: StorageData::Cuda(CudaStorage {
                        device_id: cuda.device_id(),
                        ptr: 0,
                        len: cpu.data.len(),
                    }),
                })
            }
            (StorageData::Cuda(cuda), Device::Cpu) => {
                Ok(Storage {
                    dtype: self.dtype,
                    layout: self.layout.clone(),
                    data: StorageData::Cpu(CpuStorage {
                        data: vec![0u8; cuda.len],
                    }),
                })
            }
            _ => Ok(self.clone()),
        }
    }

    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        match &self.data {
            StorageData::Cpu(cpu) => {
                let num_floats = cpu.data.len() / 4;
                let ptr = cpu.data.as_ptr() as *const f32;
                Some(unsafe { std::slice::from_raw_parts(ptr, num_floats) })
            }
            _ => None,
        }
    }

    pub fn as_mut_f32_slice(&mut self) -> Option<&mut [f32]> {
        match &mut self.data {
            StorageData::Cpu(cpu) => {
                let num_floats = cpu.data.len() / 4;
                let ptr = cpu.data.as_mut_ptr() as *mut f32;
                Some(unsafe { std::slice::from_raw_parts_mut(ptr, num_floats) })
            }
            _ => None,
        }
    }
}

unsafe impl Send for Storage {}
unsafe impl Sync for Storage {}
