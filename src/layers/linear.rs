use crate::core::{Tensor, DType, Device};
use crate::error::OxideResult;
use crate::ops::matmul;

pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, bias: bool, device: &Device) -> OxideResult<Self> {
        use crate::core::Shape;
        
        let weight = if bias {
            Tensor::from_vec(
                vec![0.01f32; in_features * out_features],
                Shape::new(&[out_features, in_features])
            )?
        } else {
            Tensor::from_vec(
                vec![0.01f32; in_features * out_features],
                Shape::new(&[out_features, in_features])
            )?
        };

        let bias_tensor = if bias {
            Some(Tensor::from_vec(
                vec![0.0f32; out_features],
                Shape::new(&[out_features])
            )?)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias: bias_tensor,
        })
    }

    pub fn from_tensors(weight: Tensor, bias: Option<Tensor>) -> OxideResult<Self> {
        Ok(Self { weight, bias })
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    pub fn forward(&self, input: &Tensor) -> OxideResult<Tensor> {
        let output = matmul(input, &self.weight.transpose(0, 1)?)?;
        
        if let Some(ref bias) = self.bias {
            let output = crate::ops::add(&output, bias)?;
            Ok(output)
        } else {
            Ok(output)
        }
    }

    pub fn in_features(&self) -> usize {
        self.weight.dims()[1]
    }

    pub fn out_features(&self) -> usize {
        self.weight.dims()[0]
    }
}
