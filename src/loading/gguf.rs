use std::collections::HashMap;
use crate::core::{Tensor, DType, Shape, Device};
use crate::error::{OxideError, OxideResult, bail};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GgmlDType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    BF16,
}

impl GgmlDType {
    pub fn from_u32(val: u32) -> OxideResult<Self> {
        match val {
            0 => Ok(GgmlDType::F32),
            1 => Ok(GgmlDType::F16),
            2 => Ok(GgmlDType::Q4_0),
            3 => Ok(GgmlDType::Q4_1),
            6 => Ok(GgmlDType::Q5_0),
            7 => Ok(GgmlDType::Q5_1),
            8 => Ok(GgmlDType::Q8_0),
            9 => Ok(GgmlDType::Q8_1),
            10 => Ok(GgmlDType::Q2K),
            11 => Ok(GgmlDType::Q3K),
            12 => Ok(GgmlDType::Q4K),
            13 => Ok(GgmlDType::Q5K),
            14 => Ok(GgmlDType::Q6K),
            15 => Ok(GgmlDType::Q8K),
            _ => bail!("Unknown GGML dtype: {}", val),
        }
    }

    pub fn block_size(&self) -> usize {
        match self {
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => 1,
            GgmlDType::Q4_0 | GgmlDType::Q4_1 | GgmlDType::Q5_0 | GgmlDType::Q5_1 => 32,
            GgmlDType::Q8_0 | GgmlDType::Q8_1 => 32,
            GgmlDType::Q2K | GgmlDType::Q3K | GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K | GgmlDType::Q8K => 256,
        }
    }

    pub fn type_size(&self) -> usize {
        match self {
            GgmlDType::F32 => 4,
            GgmlDType::F16 => 2,
            GgmlDType::BF16 => 2,
            GgmlDType::Q4_0 => 18,
            GgmlDType::Q4_1 => 20,
            GgmlDType::Q5_0 => 22,
            GgmlDType::Q5_1 => 24,
            GgmlDType::Q8_0 => 34,
            GgmlDType::Q8_1 => 36,
            GgmlDType::Q2K => 2 + 1 + 0 + 1 + 16 + 64 * 16 / 4,
            GgmlDType::Q3K => 2 + 1 + 256 / 8 + 32 * 3,
            GgmlDType::Q4K => 2 + 2 + 12 + 128,
            GgmlDType::Q5K => 2 + 2 + 12 + 160,
            GgmlDType::Q6K => 2 + 2 + 256 / 8 + 128 * 3,
            GgmlDType::Q8K => 2 + 256 + 128 * 2 + 4 * 32,
        }
    }
}

pub struct GgufMetadata {
    pub version: u32,
    pub num_tensors: u64,
    pub num_kv: u64,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<GgufTensorInfo>,
}

#[derive(Debug, Clone)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub dtype: GgmlDType,
    pub offset: u64,
}

pub struct GgufLoader;

impl GgufLoader {
    pub fn load_metadata(path: &str) -> OxideResult<GgufMetadata> {
        let data = std::fs::read(path)
            .map_err(|e| OxideError::IoError(e))?;

        Self::parse_metadata(&data)
    }

    pub fn load_tensor(
        path: &str,
        tensor_info: &GgufTensorInfo,
        device: &Device,
    ) -> OxideResult<Tensor> {
        let data = std::fs::read(path)
            .map_err(|e| OxideError::IoError(e))?;

        let tensor_offset = tensor_info.offset as usize;
        
        let dims: Vec<usize> = tensor_info.dims.iter().map(|d| *d as usize).collect();
        let shape = Shape::new(&dims);

        match tensor_info.dtype {
            GgmlDType::F32 => {
                let num_floats = shape.elem_count();
                let mut floats = Vec::with_capacity(num_floats);
                
                for i in 0..num_floats {
                    let offset = tensor_offset + i * 4;
                    let bytes = [
                        data[offset],
                        data[offset + 1],
                        data[offset + 2],
                        data[offset + 3],
                    ];
                    floats.push(f32::from_le_bytes(bytes));
                }
                
                Tensor::from_vec(floats, shape)
            }
            GgmlDType::F16 => {
                let num_floats = shape.elem_count();
                let mut floats = Vec::with_capacity(num_floats);
                
                for i in 0..num_floats {
                    let offset = tensor_offset + i * 2;
                    let val = half::f16::from_le_bytes([data[offset], data[offset + 1]]);
                    floats.push(f32::from(val));
                }
                
                Tensor::from_vec(floats, shape)
            }
            _ => {
                bail!("GGUF dtype {:?} not yet implemented", tensor_info.dtype)
            }
        }
    }

    fn parse_metadata(data: &[u8]) -> OxideResult<GgufMetadata> {
        if data.len() < 4 {
            bail!("GGUF file too small");
        }

        let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        if magic != 0x46554747 {
            bail!("Invalid GGUF magic number: 0x{:08x}", magic);
        }

        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        let num_tensors = u64::from_le_bytes([
            data[8], data[9], data[10], data[11],
            data[12], data[13], data[14], data[15],
        ]);
        let num_kv = u64::from_le_bytes([
            data[16], data[17], data[18], data[19],
            data[20], data[21], data[22], data[23],
        ]);

        let mut offset = 24;
        let mut metadata = HashMap::new();

        for _ in 0..num_kv {
            let (key, value, new_offset) = Self::parse_kv(data, offset)?;
            metadata.insert(key, value);
            offset = new_offset;
        }

        let mut tensors = Vec::with_capacity(num_tensors as usize);
        for _ in 0..num_tensors {
            let (info, new_offset) = Self::parse_tensor_info(data, offset)?;
            tensors.push(info);
            offset = new_offset;
        }

        Ok(GgufMetadata {
            version,
            num_tensors,
            num_kv,
            metadata,
            tensors,
        })
    }

    fn parse_kv(data: &[u8], mut offset: usize) -> OxideResult<(String, GgufValue, usize)> {
        let key_len = u64::from_le_bytes([
            data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
            data[offset + 4], data[offset + 5], data[offset + 6], data[offset + 7],
        ]) as usize;
        offset += 8;

        let key = String::from_utf8(data[offset..offset + key_len].to_vec())
            .map_err(|_| OxideError::InvalidArgument("Invalid UTF-8 in key".to_string()))?;
        offset += key_len;

        let dtype = u32::from_le_bytes([
            data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
        ]);
        offset += 4;

        let (value, new_offset) = Self::parse_value(data, offset, dtype)?;

        Ok((key, value, new_offset))
    }

    fn parse_tensor_info(data: &[u8], mut offset: usize) -> OxideResult<(GgufTensorInfo, usize)> {
        let name_len = u64::from_le_bytes([
            data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
            data[offset + 4], data[offset + 5], data[offset + 6], data[offset + 7],
        ]) as usize;
        offset += 8;

        let name = String::from_utf8(data[offset..offset + name_len].to_vec())
            .map_err(|_| OxideError::InvalidArgument("Invalid UTF-8 in tensor name".to_string()))?;
        offset += name_len;

        let num_dims = u32::from_le_bytes([
            data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
        ]) as usize;
        offset += 4;

        let mut dims = Vec::with_capacity(num_dims);
        for _ in 0..num_dims {
            dims.push(u64::from_le_bytes([
                data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
                data[offset + 4], data[offset + 5], data[offset + 6], data[offset + 7],
            ]));
            offset += 8;
        }

        let dtype_val = u32::from_le_bytes([
            data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
        ]);
        let dtype = GgmlDType::from_u32(dtype_val)?;
        offset += 4;

        let tensor_offset = u64::from_le_bytes([
            data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
            data[offset + 4], data[offset + 5], data[offset + 6], data[offset + 7],
        ]);
        offset += 8;

        let info = GgufTensorInfo {
            name,
            dims,
            dtype,
            offset: tensor_offset,
        };

        Ok((info, offset))
    }

    fn parse_value(data: &[u8], offset: usize, dtype: u32) -> OxideResult<(GgufValue, usize)> {
        let mut new_offset = offset;
        
        let value = match dtype {
            0 => {
                let val = data[new_offset];
                new_offset += 1;
                GgufValue::Uint8(val)
            }
            1 => {
                let val = data[new_offset] as i8;
                new_offset += 1;
                GgufValue::Int8(val)
            }
            2 => {
                let val = u16::from_le_bytes([data[new_offset], data[new_offset + 1]]);
                new_offset += 2;
                GgufValue::Uint16(val)
            }
            3 => {
                let val = i16::from_le_bytes([data[new_offset], data[new_offset + 1]]);
                new_offset += 2;
                GgufValue::Int16(val)
            }
            4 => {
                let val = u32::from_le_bytes([data[new_offset], data[new_offset + 1], data[new_offset + 2], data[new_offset + 3]]);
                new_offset += 4;
                GgufValue::Uint32(val)
            }
            5 => {
                let val = i32::from_le_bytes([data[new_offset], data[new_offset + 1], data[new_offset + 2], data[new_offset + 3]]);
                new_offset += 4;
                GgufValue::Int32(val)
            }
            6 => {
                let val = f32::from_le_bytes([data[new_offset], data[new_offset + 1], data[new_offset + 2], data[new_offset + 3]]);
                new_offset += 4;
                GgufValue::Float32(val)
            }
            7 => {
                let val = u64::from_le_bytes([
                    data[new_offset], data[new_offset + 1], data[new_offset + 2], data[new_offset + 3],
                    data[new_offset + 4], data[new_offset + 5], data[new_offset + 6], data[new_offset + 7],
                ]);
                new_offset += 8;
                GgufValue::Uint64(val)
            }
            8 => {
                let val = i64::from_le_bytes([
                    data[new_offset], data[new_offset + 1], data[new_offset + 2], data[new_offset + 3],
                    data[new_offset + 4], data[new_offset + 5], data[new_offset + 6], data[new_offset + 7],
                ]);
                new_offset += 8;
                GgufValue::Int64(val)
            }
            9 => {
                let val = f64::from_le_bytes([
                    data[new_offset], data[new_offset + 1], data[new_offset + 2], data[new_offset + 3],
                    data[new_offset + 4], data[new_offset + 5], data[new_offset + 6], data[new_offset + 7],
                ]);
                new_offset += 8;
                GgufValue::Float64(val)
            }
            10 => {
                let val = data[new_offset] != 0;
                new_offset += 1;
                GgufValue::Bool(val)
            }
            11 => {
                let len = u64::from_le_bytes([
                    data[new_offset], data[new_offset + 1], data[new_offset + 2], data[new_offset + 3],
                    data[new_offset + 4], data[new_offset + 5], data[new_offset + 6], data[new_offset + 7],
                ]) as usize;
                new_offset += 8;
                let val = String::from_utf8(data[new_offset..new_offset + len].to_vec())
                    .map_err(|_| OxideError::InvalidArgument("Invalid UTF-8 in string value".to_string()))?;
                new_offset += len;
                GgufValue::String(val)
            }
            _ => bail!("Unknown GGUF value type: {}", dtype),
        };

        Ok((value, new_offset))
    }
}

pub fn load_gguf_metadata(path: &str) -> OxideResult<GgufMetadata> {
    GgufLoader::load_metadata(path)
}

pub fn load_gguf_tensor(
    path: &str,
    tensor_info: &GgufTensorInfo,
    device: &Device,
) -> OxideResult<Tensor> {
    GgufLoader::load_tensor(path, tensor_info, device)
}
