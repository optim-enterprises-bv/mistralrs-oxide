use crate::core::{Shape, Tensor, DType, Storage};
use crate::error::OxideResult;

pub fn embedding(indices: &Tensor, weight: &Tensor) -> OxideResult<Tensor> {
    let indices_dims = indices.dims();
    let weight_dims = weight.dims();

    if weight_dims.len() != 2 {
        return Err(crate::error::OxideError::InvalidArgument(
            "embedding weight must be 2D".to_string()
        ));
    }

    let vocab_size = weight_dims[0];
    let hidden_dim = weight_dims[1];

    let mut output_shape: Vec<usize> = indices_dims.to_vec();
    output_shape.push(hidden_dim);

    let output = Tensor::zeros(Shape::new(&output_shape), weight.dtype(), &weight.device())?;

    if weight.device().is_cpu() {
        let indices_data: Vec<i64> = indices.to_vec1()
            .map_err(|_| crate::error::OxideError::InvalidArgument("Expected i64 indices".to_string()))?;
        let weight_data = weight.storage().as_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32 weight".to_string()))?;
        let out_data = (*output.storage()).as_mut_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32 output".to_string()))?;

        let num_indices = indices.elem_count();
        
        for i in 0..num_indices {
            let idx = indices_data[i] as usize;
            if idx >= vocab_size {
                return Err(crate::error::OxideError::InvalidArgument(
                    format!("embedding index {} out of bounds for vocab size {}", idx, vocab_size)
                ));
            }
            
            let weight_offset = idx * hidden_dim;
            let out_offset = i * hidden_dim;
            
            for j in 0..hidden_dim {
                out_data[out_offset + j] = weight_data[weight_offset + j];
            }
        }
    }

    Ok(output)
}
