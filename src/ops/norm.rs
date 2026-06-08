use crate::core::{Shape, Tensor, DType, Storage};
use crate::error::OxideResult;

pub fn rms_norm(input: &Tensor, weight: &Tensor, eps: f32) -> OxideResult<Tensor> {
    let dims = input.dims();
    if dims.len() < 1 {
        return Err(crate::error::OxideError::InvalidArgument(
            "rms_norm requires at least 1D tensor".to_string()
        ));
    }

    let last_dim = dims[dims.len() - 1];
    let weight_dims = weight.dims();
    if weight_dims.len() != 1 || weight_dims[0] != last_dim {
        return Err(crate::error::OxideError::InvalidArgument(format!(
            "rms_norm weight shape {:?} doesn't match input last dim {}",
            weight_dims, last_dim
        )));
    }

    let mut output = Tensor::zeros(input.shape().clone(), input.dtype(), &input.device())?;
    
    if input.device().is_cpu() {
        let input_data = input.storage().as_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32".to_string()))?;
        let weight_data = weight.storage().as_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32".to_string()))?;
        let out_data = (*output.storage()).as_mut_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32".to_string()))?;
        
        let num_rows = input.elem_count() / last_dim;
        
        for row in 0..num_rows {
            let row_offset = row * last_dim;
            
            let mut sum_squares = 0.0f32;
            for i in 0..last_dim {
                let val = input_data[row_offset + i];
                sum_squares += val * val;
            }
            let rms = (sum_squares / last_dim as f32 + eps).sqrt();
            
            for i in 0..last_dim {
                let normalized = input_data[row_offset + i] / rms;
                out_data[row_offset + i] = normalized * weight_data[i];
            }
        }
    }
    
    Ok(output)
}

pub fn layer_norm(input: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32) -> OxideResult<Tensor> {
    let dims = input.dims();
    if dims.len() < 1 {
        return Err(crate::error::OxideError::InvalidArgument(
            "layer_norm requires at least 1D tensor".to_string()
        ));
    }

    let last_dim = dims[dims.len() - 1];
    let weight_dims = weight.dims();
    let bias_dims = bias.dims();
    
    if weight_dims.len() != 1 || weight_dims[0] != last_dim {
        return Err(crate::error::OxideError::InvalidArgument(format!(
            "layer_norm weight shape {:?} doesn't match input last dim {}",
            weight_dims, last_dim
        )));
    }
    if bias_dims.len() != 1 || bias_dims[0] != last_dim {
        return Err(crate::error::OxideError::InvalidArgument(format!(
            "layer_norm bias shape {:?} doesn't match input last dim {}",
            bias_dims, last_dim
        )));
    }

    let mut output = Tensor::zeros(input.shape().clone(), input.dtype(), &input.device())?;
    
    if input.device().is_cpu() {
        let input_data = input.storage().as_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32".to_string()))?;
        let weight_data = weight.storage().as_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32".to_string()))?;
        let bias_data = bias.storage().as_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32".to_string()))?;
        let out_data = (*output.storage()).as_mut_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32".to_string()))?;
        
        let num_rows = input.elem_count() / last_dim;
        
        for row in 0..num_rows {
            let row_offset = row * last_dim;
            
            let mut sum = 0.0f32;
            for i in 0..last_dim {
                sum += input_data[row_offset + i];
            }
            let mean = sum / last_dim as f32;
            
            let mut sum_sq = 0.0f32;
            for i in 0..last_dim {
                let diff = input_data[row_offset + i] - mean;
                sum_sq += diff * diff;
            }
            let var = sum_sq / last_dim as f32;
            let std = (var + eps).sqrt();
            
            for i in 0..last_dim {
                let normalized = (input_data[row_offset + i] - mean) / std;
                out_data[row_offset + i] = normalized * weight_data[i] + bias_data[i];
            }
        }
    }
    
    Ok(output)
}
