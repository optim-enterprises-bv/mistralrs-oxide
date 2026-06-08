use crate::core::{Shape, Tensor, DType, Storage};
use crate::error::OxideResult;

pub fn exp(x: &Tensor) -> OxideResult<Tensor> {
    let mut output = Tensor::zeros(x.shape().clone(), x.dtype(), &x.device())?;
    
    if x.device().is_cpu() {
        let x_data = x.storage().as_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32".to_string()))?;
        let out_data = (*output.storage()).as_mut_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32".to_string()))?;
        
        for i in 0..x_data.len() {
            out_data[i] = x_data[i].exp();
        }
    }
    
    Ok(output)
}

pub fn log(x: &Tensor) -> OxideResult<Tensor> {
    let mut output = Tensor::zeros(x.shape().clone(), x.dtype(), &x.device())?;
    
    if x.device().is_cpu() {
        let x_data = x.storage().as_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32".to_string()))?;
        let out_data = (*output.storage()).as_mut_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32".to_string()))?;
        
        for i in 0..x_data.len() {
            out_data[i] = x_data[i].ln();
        }
    }
    
    Ok(output)
}

pub fn sqrt(x: &Tensor) -> OxideResult<Tensor> {
    let mut output = Tensor::zeros(x.shape().clone(), x.dtype(), &x.device())?;
    
    if x.device().is_cpu() {
        let x_data = x.storage().as_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32".to_string()))?;
        let out_data = (*output.storage()).as_mut_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32".to_string()))?;
        
        for i in 0..x_data.len() {
            out_data[i] = x_data[i].sqrt();
        }
    }
    
    Ok(output)
}

pub fn pow(x: &Tensor, n: f32) -> OxideResult<Tensor> {
    let mut output = Tensor::zeros(x.shape().clone(), x.dtype(), &x.device())?;
    
    if x.device().is_cpu() {
        let x_data = x.storage().as_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32".to_string()))?;
        let out_data = (*output.storage()).as_mut_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32".to_string()))?;
        
        for i in 0..x_data.len() {
            out_data[i] = x_data[i].powf(n);
        }
    }
    
    Ok(output)
}

pub fn clamp(x: &Tensor, min: f32, max: f32) -> OxideResult<Tensor> {
    let mut output = Tensor::zeros(x.shape().clone(), x.dtype(), &x.device())?;
    
    if x.device().is_cpu() {
        let x_data = x.storage().as_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32".to_string()))?;
        let out_data = (*output.storage()).as_mut_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32".to_string()))?;
        
        for i in 0..x_data.len() {
            out_data[i] = x_data[i].clamp(min, max);
        }
    }
    
    Ok(output)
}

pub fn neg(x: &Tensor) -> OxideResult<Tensor> {
    let mut output = Tensor::zeros(x.shape().clone(), x.dtype(), &x.device())?;
    
    if x.device().is_cpu() {
        let x_data = x.storage().as_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32".to_string()))?;
        let out_data = (*output.storage()).as_mut_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32".to_string()))?;
        
        for i in 0..x_data.len() {
            out_data[i] = -x_data[i];
        }
    }
    
    Ok(output)
}

pub fn abs(x: &Tensor) -> OxideResult<Tensor> {
    let mut output = Tensor::zeros(x.shape().clone(), x.dtype(), &x.device())?;
    
    if x.device().is_cpu() {
        let x_data = x.storage().as_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32".to_string()))?;
        let out_data = (*output.storage()).as_mut_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32".to_string()))?;
        
        for i in 0..x_data.len() {
            out_data[i] = x_data[i].abs();
        }
    }
    
    Ok(output)
}
