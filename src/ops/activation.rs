use crate::core::{Shape, Tensor, DType, Storage};
use crate::error::OxideResult;

pub fn relu(input: &Tensor) -> OxideResult<Tensor> {
    let mut output = Tensor::zeros(input.shape().clone(), input.dtype(), &input.device())?;
    
    if input.device().is_cpu() {
        let input_data = input.storage().as_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32".to_string()))?;
        let out_data = (*output.storage()).as_mut_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32".to_string()))?;
        
        for i in 0..input_data.len() {
            out_data[i] = input_data[i].max(0.0);
        }
    }
    
    Ok(output)
}

pub fn silu(input: &Tensor) -> OxideResult<Tensor> {
    let mut output = Tensor::zeros(input.shape().clone(), input.dtype(), &input.device())?;
    
    if input.device().is_cpu() {
        let input_data = input.storage().as_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32".to_string()))?;
        let out_data = (*output.storage()).as_mut_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32".to_string()))?;
        
        for i in 0..input_data.len() {
            let x = input_data[i];
            out_data[i] = x * (1.0 / (1.0 + (-x).exp()));
        }
    }
    
    Ok(output)
}

pub fn gelu(input: &Tensor) -> OxideResult<Tensor> {
    let mut output = Tensor::zeros(input.shape().clone(), input.dtype(), &input.device())?;
    
    if input.device().is_cpu() {
        use std::f32::consts::PI;
        let input_data = input.storage().as_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32".to_string()))?;
        let out_data = (*output.storage()).as_mut_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32".to_string()))?;
        
        for i in 0..input_data.len() {
            let x = input_data[i];
            out_data[i] = 0.5 * x * (1.0 + ((2.0 / PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh());
        }
    }
    
    Ok(output)
}

pub fn gelu_quick(input: &Tensor) -> OxideResult<Tensor> {
    let mut output = Tensor::zeros(input.shape().clone(), input.dtype(), &input.device())?;
    
    if input.device().is_cpu() {
        let input_data = input.storage().as_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32".to_string()))?;
        let out_data = (*output.storage()).as_mut_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32".to_string()))?;
        
        for i in 0..input_data.len() {
            let x = input_data[i];
            out_data[i] = x * (1.0 / (1.0 + (-1.702 * x).exp()));
        }
    }
    
    Ok(output)
}

pub fn softmax(input: &Tensor, dim: usize) -> OxideResult<Tensor> {
    if dim >= input.rank() {
        return Err(crate::error::OxideError::InvalidArgument(format!(
            "softmax dim {} out of range for rank {}", dim, input.rank()
        )));
    }

    let mut output = Tensor::zeros(input.shape().clone(), input.dtype(), &input.device())?;
    
    if input.device().is_cpu() {
        let input_data = input.storage().as_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32".to_string()))?;
        let out_data = (*output.storage()).as_mut_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32".to_string()))?;
        
        let dims = input.dims();
        let dim_size = dims[dim];
        let outer_stride: usize = dims[dim..].iter().product();
        let inner_stride: usize = dims[dim+1..].iter().product();
        let outer_count = input.elem_count() / outer_stride;
        
        for batch in 0..outer_count {
            let base_idx = batch * outer_stride;
            
            for inner in 0..inner_stride {
                let mut max_val = f32::NEG_INFINITY;
                for d in 0..dim_size {
                    let idx = base_idx + d * inner_stride + inner;
                    max_val = max_val.max(input_data[idx]);
                }
                
                let mut sum = 0.0f32;
                for d in 0..dim_size {
                    let idx = base_idx + d * inner_stride + inner;
                    let exp_val = (input_data[idx] - max_val).exp();
                    out_data[idx] = exp_val;
                    sum += exp_val;
                }
                
                for d in 0..dim_size {
                    let idx = base_idx + d * inner_stride + inner;
                    out_data[idx] /= sum;
                }
            }
        }
    }
    
    Ok(output)
}

pub fn sigmoid(input: &Tensor) -> OxideResult<Tensor> {
    let mut output = Tensor::zeros(input.shape().clone(), input.dtype(), &input.device())?;
    
    if input.device().is_cpu() {
        let input_data = input.storage().as_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32".to_string()))?;
        let out_data = (*output.storage()).as_mut_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32".to_string()))?;
        
        for i in 0..input_data.len() {
            out_data[i] = 1.0 / (1.0 + (-input_data[i]).exp());
        }
    }
    
    Ok(output)
}

pub fn tanh(input: &Tensor) -> OxideResult<Tensor> {
    let mut output = Tensor::zeros(input.shape().clone(), input.dtype(), &input.device())?;
    
    if input.device().is_cpu() {
        let input_data = input.storage().as_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32".to_string()))?;
        let out_data = (*output.storage()).as_mut_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32".to_string()))?;
        
        for i in 0..input_data.len() {
            out_data[i] = input_data[i].tanh();
        }
    }
    
    Ok(output)
}
