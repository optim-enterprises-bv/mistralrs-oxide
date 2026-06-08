use std::sync::Arc;
use crate::core::{Device, DType, Shape, Layout, Storage, StorageData, CpuStorage};
use crate::error::{OxideError, OxideResult, bail};

#[derive(Clone)]
pub struct Tensor {
    storage: Arc<Storage>,
}

impl Tensor {
    pub fn from_vec(data: Vec<f32>, shape: Shape) -> OxideResult<Self> {
        let layout = Layout::contiguous(shape);
        let num_bytes = data.len() * 4;
        let bytes: Vec<u8> = data.iter()
            .flat_map(|f| f.to_ne_bytes())
            .collect();
        
        assert_eq!(bytes.len(), num_bytes);

        let storage = Arc::new(Storage {
            dtype: DType::F32,
            layout,
            data: StorageData::Cpu(CpuStorage { data: bytes }),
        });

        Ok(Self { storage })
    }

    pub fn zeros(shape: Shape, dtype: DType, device: &Device) -> OxideResult<Self> {
        let layout = Layout::contiguous(shape.clone());
        let num_bytes = shape.elem_count() * dtype.size_in_bytes();

        let data = match device {
            Device::Cpu => {
                StorageData::Cpu(CpuStorage { data: vec![0u8; num_bytes] })
            }
            Device::Cuda(cuda) => {
                StorageData::Cuda(crate::core::CudaStorage {
                    device_id: cuda.device_id(),
                    ptr: 0,
                    len: num_bytes,
                })
            }
        };

        Ok(Self {
            storage: Arc::new(Storage {
                dtype,
                layout,
                data,
            }),
        })
    }

    pub fn ones(shape: Shape, dtype: DType, device: &Device) -> OxideResult<Self> {
        let mut tensor = Self::zeros(shape, dtype, device)?;
        
        if let Some(slice) = Arc::make_mut(&mut Arc::clone(&tensor.storage))
            .as_mut_f32_slice() {
            slice.fill(1.0);
        }

        Ok(tensor)
    }

    pub fn shape(&self) -> &Shape {
        &self.storage.layout.shape
    }

    pub fn dtype(&self) -> DType {
        self.storage.dtype
    }

    pub fn device(&self) -> Device {
        self.storage.device()
    }

    pub fn dims(&self) -> &[usize] {
        self.shape().dims()
    }

    pub fn rank(&self) -> usize {
        self.shape().rank()
    }

    pub fn elem_count(&self) -> usize {
        self.shape().elem_count()
    }

    pub fn is_contiguous(&self) -> bool {
        self.storage.is_contiguous()
    }

    pub fn to_device(&self, device: &Device) -> OxideResult<Tensor> {
        let new_storage = self.storage.to_device(device)?;
        Ok(Tensor::from_storage(new_storage))
    }

    pub fn reshape(&self, new_shape: Shape) -> OxideResult<Tensor> {
        if self.shape().elem_count() != new_shape.elem_count() {
            bail!("reshape: shape mismatch {} != {}", 
                self.shape().elem_count(), new_shape.elem_count());
        }

        let mut storage = (*self.storage).clone();
        storage.layout = Layout::contiguous(new_shape);
        
        Ok(Tensor::from_storage(storage))
    }

    pub fn transpose(&self, dim0: usize, dim1: usize) -> OxideResult<Tensor> {
        let rank = self.rank();
        if dim0 >= rank || dim1 >= rank {
            bail!("transpose: dims {} and {} out of range for rank {}", dim0, dim1, rank);
        }

        let mut new_dims = self.dims().to_vec();
        new_dims.swap(dim0, dim1);
        let new_shape = Shape::new(&new_dims);

        let mut new_strides = self.storage.layout.strides[..rank].to_vec();
        new_strides.swap(dim0, dim1);

        let new_layout = Layout::new(new_shape, new_strides, self.storage.layout.offset);

        let mut new_storage = (*self.storage).clone();
        new_storage.layout = new_layout;

        Ok(Tensor::from_storage(new_storage))
    }

    pub fn to_vec1<T: Copy>(&self) -> OxideResult<Vec<T>> {
        let cpu_storage = if self.device().is_cuda() {
            self.storage.to_device(&Device::Cpu)?
        } else {
            (*self.storage).clone()
        };

        match &cpu_storage.data {
            StorageData::Cpu(cpu) => {
                let count = self.elem_count();
                let size = std::mem::size_of::<T>();
                if cpu.data.len() != count * size {
                    bail!("Size mismatch: expected {} bytes, got {}", 
                        count * size, cpu.data.len());
                }
                
                let mut result = Vec::with_capacity(count);
                for i in 0..count {
                    let offset = i * size;
                    let ptr = cpu.data[offset..offset + size].as_ptr() as *const T;
                    result.push(unsafe { *ptr });
                }
                Ok(result)
            }
            _ => unreachable!(),
        }
    }

    pub fn to_f32_vec(&self) -> OxideResult<Vec<f32>> {
        self.to_vec1::<f32>()
    }

    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> OxideResult<Tensor> {
        let dims = self.dims();
        if dim >= dims.len() {
            bail!("narrow: dim {} out of range for {} dimensions", dim, dims.len());
        }
        if start + len > dims[dim] {
            bail!("narrow: range {}..{} exceeds dimension size {}",
                start, start + len, dims[dim]);
        }

        let new_layout = self.storage.layout.narrow(dim, start, len);

        let new_storage = Storage {
            dtype: self.storage.dtype,
            layout: new_layout,
            data: match &self.storage.data {
                StorageData::Cpu(c) => StorageData::Cpu(CpuStorage { data: c.data.clone() }),
                StorageData::Cuda(c) => StorageData::Cuda(c.clone()),
            },
        };

        Ok(Tensor::from_storage(new_storage))
    }

    pub fn cat(tensors: &[Tensor], dim: usize) -> OxideResult<Tensor> {
        if tensors.is_empty() {
            bail!("cat: requires at least one tensor");
        }

        let first = &tensors[0];
        let rank = first.rank();
        
        if dim >= rank {
            bail!("cat: dim {} out of range for rank {}", dim, rank);
        }

        let dtype = first.dtype();
        let device = first.device();

        let mut concat_dim_size = 0;
        for t in tensors {
            if t.dtype() != dtype {
                bail!("cat: dtype mismatch {:?} != {:?}", t.dtype(), dtype);
            }
            if t.rank() != rank {
                bail!("cat: rank mismatch {} != {}", t.rank(), rank);
            }
            concat_dim_size += t.dims()[dim];
        }

        let mut new_dims = first.dims().to_vec();
        new_dims[dim] = concat_dim_size;
        let new_shape = Shape::new(&new_dims);

        let result = Tensor::zeros(new_shape, dtype, &device)?;

        Ok(result)
    }

    pub fn from_storage(storage: Storage) -> Self {
        Self {
            storage: Arc::new(storage),
        }
    }

    pub fn storage(&self) -> &Storage {
        &self.storage
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", self.shape())
            .field("dtype", &self.dtype())
            .field("device", &self.device())
            .finish()
    }
}
