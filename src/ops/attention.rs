use crate::core::{Shape, Tensor, DType, Storage};
use crate::error::OxideResult;

pub fn scaled_dot_product_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    mask: Option<&Tensor>,
) -> OxideResult<Tensor> {
    let q_shape = query.dims();
    let k_shape = key.dims();
    let v_shape = value.dims();

    if q_shape.len() < 3 || k_shape.len() < 3 || v_shape.len() < 3 {
        return Err(crate::error::OxideError::InvalidArgument(
            "Attention requires 3D+ tensors (batch, seq_len, dim)".to_string()
        ));
    }

    let batch_size = q_shape[0];
    let num_heads = q_shape[1];
    let q_len = q_shape[2];
    let head_dim = q_shape[q_shape.len() - 1];
    let kv_len = k_shape[2];

    let mut output_shape = q_shape.to_vec();
    output_shape[output_shape.len() - 1] = v_shape[v_shape.len() - 1];
    
    let output = Tensor::zeros(Shape::new(&output_shape), query.dtype(), &query.device())?;

    if query.device().is_cpu() {
        let q_data = query.storage().as_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32".to_string()))?;
        let k_data = key.storage().as_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32".to_string()))?;
        let v_data = value.storage().as_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32".to_string()))?;
        let out_data = (*output.storage()).as_mut_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32".to_string()))?;

        let scale = (head_dim as f32).sqrt().recip();
        let v_head_dim = v_shape[v_shape.len() - 1];

        for b in 0..batch_size {
            for h in 0..num_heads {
                let q_offset = ((b * num_heads + h) * q_len) * head_dim;
                let k_offset = ((b * num_heads + h) * kv_len) * head_dim;
                let v_offset = ((b * num_heads + h) * kv_len) * v_head_dim;
                let out_offset = ((b * num_heads + h) * q_len) * v_head_dim;

                for i in 0..q_len {
                    let mut attn_scores = vec![0.0f32; kv_len];
                    
                    for j in 0..kv_len {
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q_data[q_offset + i * head_dim + d] * 
                                   k_data[k_offset + j * head_dim + d];
                        }
                        attn_scores[j] = dot * scale;
                    }

                    if let Some(m) = mask {
                        let mask_data = m.storage().as_f32_slice()
                            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32".to_string()))?;
                        for j in 0..kv_len {
                            attn_scores[j] += mask_data[j % mask.elem_count()];
                        }
                    }

                    let max_score = attn_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum_exp = 0.0f32;
                    for j in 0..kv_len {
                        attn_scores[j] = (attn_scores[j] - max_score).exp();
                        sum_exp += attn_scores[j];
                    }
                    for j in 0..kv_len {
                        attn_scores[j] /= sum_exp;
                    }

                    for d in 0..v_head_dim {
                        let mut weighted_sum = 0.0f32;
                        for j in 0..kv_len {
                            weighted_sum += attn_scores[j] * v_data[v_offset + j * v_head_dim + d];
                        }
                        out_data[out_offset + i * v_head_dim + d] = weighted_sum;
                    }
                }
            }
        }
    }

    Ok(output)
}

pub fn apply_rotary_emb(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> OxideResult<Tensor> {
    let x_dims = x.dims();
    if x_dims.len() < 3 {
        return Err(crate::error::OxideError::InvalidArgument(
            "apply_rotary_emb requires at least 3D tensor".to_string()
        ));
    }

    let seq_len = x_dims[x_dims.len() - 2];
    let head_dim = x_dims[x_dims.len() - 1];
    let half_dim = head_dim / 2;

    if cos.dims() != &[seq_len, half_dim][..] {
        return Err(crate::error::OxideError::InvalidArgument(format!(
            "cos shape {:?} doesn't match expected {:?}",
            cos.dims(), [seq_len, half_dim]
        )));
    }
    if sin.dims() != &[seq_len, half_dim][..] {
        return Err(crate::error::OxideError::InvalidArgument(format!(
            "sin shape {:?} doesn't match expected {:?}",
            sin.dims(), [seq_len, half_dim]
        )));
    }

    let mut output = Tensor::zeros(x.shape().clone(), x.dtype(), &x.device())?;

    if x.device().is_cpu() {
        let x_data = x.storage().as_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32".to_string()))?;
        let cos_data = cos.storage().as_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32".to_string()))?;
        let sin_data = sin.storage().as_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32".to_string()))?;
        let out_data = (*output.storage()).as_mut_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32".to_string()))?;

        let num_heads = x.elem_count() / (seq_len * head_dim);

        for h in 0..num_heads {
            let base_offset = h * seq_len * head_dim;
            
            for pos in 0..seq_len {
                let head_offset = base_offset + pos * head_dim;
                
                for d in 0..half_dim {
                    let x1 = x_data[head_offset + d];
                    let x2 = x_data[head_offset + d + half_dim];
                    let c = cos_data[pos * half_dim + d];
                    let s = sin_data[pos * half_dim + d];
                    
                    out_data[head_offset + d] = x1 * c - x2 * s;
                    out_data[head_offset + d + half_dim] = x1 * s + x2 * c;
                }
            }
        }
    }

    Ok(output)
}

pub fn repeat_kv(x: &Tensor, num_repeat: usize) -> OxideResult<Tensor> {
    let dims = x.dims();
    if dims.len() < 4 {
        return Err(crate::error::OxideError::InvalidArgument(
            "repeat_kv requires 4D tensor (batch, heads, seq_len, head_dim)".to_string()
        ));
    }

    let mut new_dims = dims.to_vec();
    new_dims[1] *= num_repeat;

    let output = Tensor::zeros(Shape::new(&new_dims), x.dtype(), &x.device())?;

    if x.device().is_cpu() {
        let x_data = x.storage().as_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32".to_string()))?;
        let out_data = (*output.storage()).as_mut_f32_slice()
            .ok_or_else(|| crate::error::OxideError::InvalidArgument("Expected f32".to_string()))?;

        let batch = dims[0];
        let num_heads = dims[1];
        let seq_len = dims[2];
        let head_dim = dims[3];
        let elem_per_head = seq_len * head_dim;

        for b in 0..batch {
            for h in 0..num_heads {
                for r in 0..num_repeat {
                    let src_offset = ((b * num_heads + h) * elem_per_head) as usize;
                    let dst_offset = ((b * (num_heads * num_repeat) + h * num_repeat + r) * elem_per_head) as usize;
                    
                    for i in 0..elem_per_head {
                        out_data[dst_offset + i] = x_data[src_offset + i];
                    }
                }
            }
        }
    }

    Ok(output)
}
