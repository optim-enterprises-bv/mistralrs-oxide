use std::collections::HashMap;
use crate::core::{Tensor, DType, Shape, Device};
use crate::error::{OxideError, OxideResult, bail};

pub struct SafetensorsLoader;

impl SafetensorsLoader {
    pub fn load_from_file(path: &str, device: &Device) -> OxideResult<HashMap<String, Tensor>> {
        let data = std::fs::read(path)
            .map_err(|e| OxideError::IoError(e))?;
        
        Self::load_from_bytes(&data, device)
    }

    pub fn load_from_bytes(data: &[u8], device: &Device) -> OxideResult<HashMap<String, Tensor>> {
        let tensors = safetensors::SafeTensors::deserialize(data)
            .map_err(|e| OxideError::SerializationError(e.to_string()))?;

        let mut result = HashMap::new();

        for (name, tensor_view) in tensors.iter() {
            let dtype = Self::convert_dtype(tensor_view.dtype())?;
            let shape = Shape::new(tensor_view.shape());
            
            let tensor = Self::load_tensor_from_view(&tensor_view, shape, dtype, device)?;
            result.insert(name.to_string(), tensor);
        }

        Ok(result)
    }

    fn convert_dtype(dtype: safetensors::Dtype) -> OxideResult<DType> {
        match dtype {
            safetensors::Dtype::F32 => Ok(DType::F32),
            safetensors::Dtype::F16 => Ok(DType::F16),
            safetensors::Dtype::BF16 => Ok(DType::BF16),
            safetensors::Dtype::I64 => Ok(DType::I64),
            safetensors::Dtype::U8 => Ok(DType::U8),
            safetensors::Dtype::U32 => Ok(DType::U32),
            safetensors::Dtype::F64 => Ok(DType::F64),
            safetensors::Dtype::F8_E4M3 => Ok(DType::F8E4M3),
            _ => Err(OxideError::UnsupportedOperation(
                format!("Unsupported dtype in safetensors: {:?}", dtype)
            )),
        }
    }

    fn load_tensor_from_view(
        view: &safetensors::TensorView,
        shape: Shape,
        dtype: DType,
        device: &Device,
    ) -> OxideResult<Tensor> {
        let data = view.data();
        
        match dtype {
            DType::F32 => {
                let num_floats = data.len() / 4;
                let mut floats = Vec::with_capacity(num_floats);
                
                for chunk in data.chunks_exact(4) {
                    let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
                    let val = f32::from_le_bytes(bytes);
                    floats.push(val);
                }
                
                Tensor::from_vec(floats, shape)
            }
            DType::F16 | DType::BF16 => {
                let num_floats = data.len() / 2;
                let mut floats = Vec::with_capacity(num_floats);
                
                for chunk in data.chunks_exact(2) {
                    let val = half::f16::from_le_bytes([chunk[0], chunk[1]]);
                    floats.push(f32::from(val));
                }
                
                Tensor::from_vec(floats, shape)
            }
            _ => {
                bail!("Dtype {:?} not yet implemented for loading", dtype)
            }
        }
    }

    pub fn save_to_file(
        tensors: &HashMap<String, Tensor>,
        path: &str,
    ) -> OxideResult<()> {
        let mut metadata: HashMap<String, safetensors::TensorView> = HashMap::new();
        let mut buffer: Vec<u8> = Vec::new();

        for (name, tensor) in tensors {
            let data = Self::tensor_to_bytes(tensor)?;
            let view = safetensors::TensorView::new(
                Self::convert_to_safetensors_dtype(tensor.dtype()),
                tensor.dims().to_vec(),
                buffer.len()..buffer.len() + data.len(),
            ).map_err(|e| OxideError::SerializationError(e.to_string()))?;
            
            metadata.insert(name.clone(), view);
            buffer.extend_from_slice(&data);
        }

        let st = safetensors::SafeTensors::serialize(&metadata, &buffer)
            .map_err(|e| OxideError::SerializationError(e.to_string()))?;

        std::fs::write(path, st)
            .map_err(|e| OxideError::IoError(e))?;

        Ok(())
    }

    fn tensor_to_bytes(tensor: &Tensor) -> OxideResult<Vec<u8>> {
        let cpu_tensor = if tensor.device().is_cuda() {
            tensor.to_device(&Device::Cpu)?
        } else {
            tensor.clone()
        };

        match tensor.dtype() {
            DType::F32 => {
                let floats = cpu_tensor.to_f32_vec()?;
                let mut bytes = Vec::with_capacity(floats.len() * 4);
                for f in floats {
                    bytes.extend_from_slice(&f.to_le_bytes());
                }
                Ok(bytes)
            }
            _ => Err(OxideError::UnsupportedOperation(
                format!("Save not implemented for dtype {:?}", tensor.dtype())
            )),
        }
    }

    fn convert_to_safetensors_dtype(dtype: DType) -> safetensors::Dtype {
        match dtype {
            DType::F32 => safetensors::Dtype::F32,
            DType::F16 => safetensors::Dtype::F16,
            DType::BF16 => safetensors::Dtype::BF16,
            DType::I64 => safetensors::Dtype::I64,
            DType::U8 => safetensors::Dtype::U8,
            DType::U32 => safetensors::Dtype::U32,
            DType::F64 => safetensors::Dtype::F64,
            DType::F8E4M3 => safetensors::Dtype::F8_E4M3,
        }
    }
}

pub fn load_safetensors(path: &str, device: &Device) -> OxideResult<HashMap<String, Tensor>> {
    SafetensorsLoader::load_from_file(path, device)
}
