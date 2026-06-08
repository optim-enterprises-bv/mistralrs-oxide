use crate::core::{Tensor, DType, Device, Shape};
use crate::error::OxideResult;
use crate::ops::{rms_norm, layer_norm};

pub struct RMSNorm {
    weight: Tensor,
    eps: f32,
}

impl RMSNorm {
    pub fn new(dim: usize, eps: f32, device: &Device) -> OxideResult<Self> {
        let weight = Tensor::from_vec(
            vec![1.0f32; dim],
            Shape::new(&[dim])
        )?;

        Ok(Self { weight, eps })
    }

    pub fn from_tensor(weight: Tensor, eps: f32) -> OxideResult<Self> {
        Ok(Self { weight, eps })
    }

    pub fn weight(&self) -&Tensor {
        &self.weight
    }

    pub fn forward(&self, input: &Tensor) -> OxideResult<Tensor> {
        rms_norm(input, &self.weight, self.eps)
    }

    pub fn eps(&self) -> f32 {
        self.eps
    }
}

pub struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f32,
}

impl LayerNorm {
    pub fn new(dim: usize, eps: f32, device: &Device) -> OxideResult<Self> {
        let weight = Tensor::from_vec(
            vec![1.0f32; dim],
            Shape::new(&[dim])
        )?;
        let bias = Tensor::from_vec(
            vec![0.0f32; dim],
            Shape::new(&[dim])
        )?;

        Ok(Self { weight, bias, eps })
    }

    pub fn from_tensors(weight: Tensor, bias: Tensor, eps: f32) -> OxideResult<Self> {
        Ok(Self { weight, bias, eps })
    }

    pub fn forward(&self, input: &Tensor) -> OxideResult<Tensor> {
        layer_norm(input, &self.weight, &self.bias, self.eps)
    }
}
