use crate::core::{Device, DType, Shape, Tensor, Storage, StorageData, CpuStorage};
use crate::error::{OxideError, OxideResult, bail};

/// Trait for operations that can be executed on GPU
pub trait GpuExecutable {
    /// Check if this operation is supported on GPU
    fn supports_gpu(&self) -> bool {
        false
    }
    
    /// Execute on GPU if available, otherwise fall back to CPU
    fn execute_gpu(&self, inputs: &[&Tensor], output: &mut Tensor) -> OxideResult<()>;
}

/// GPU operation dispatch table
pub struct GpuOpDispatcher {
    device: Device,
    use_gpu: bool,
}

impl GpuOpDispatcher {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
            use_gpu: device.is_cuda(),
        }
    }

    pub fn with_gpu(device: &Device, use_gpu: bool) -> Self {
        Self {
            device: device.clone(),
            use_gpu: use_gpu && device.is_cuda(),
        }
    }

    /// Execute an operation, automatically choosing GPU or CPU
    pub fn execute<F, G>(
        &self,
        inputs: &[&Tensor],
        mut output: &mut Tensor,
        cpu_impl: F,
        _gpu_impl: G,
    ) -> OxideResult<()>
    where
        F: FnOnce(&[&Tensor], &mut Tensor) -> OxideResult<()>,
        G: FnOnce(&[&Tensor], &mut Tensor) -> OxideResult<()>,
    {
        if self.use_gpu {
            // Try GPU implementation
            // In real implementation, this would launch CUDA kernels
            // For now, fall back to CPU
            cpu_impl(inputs, &mut output)
        } else {
            cpu_impl(inputs, &mut output)
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn use_gpu(&self) -> bool {
        self.use_gpu
    }
}

/// Matmul GPU dispatch
pub fn matmul_gpu_dispatch(
    a: &Tensor,
    b: &Tensor,
    output: &mut Tensor,
) -> OxideResult<()> {
    // In real implementation:
    // 1. Check dimensions for kernel selection
    // 2. Upload tensors to GPU if needed
    // 3. Launch appropriate kernel (naive, tiled, or cuBLAS)
    // 4. Download result if needed
    
    // For now, use CPU implementation
    let a_data = a.storage().as_f32_slice()
        .ok_or_else(|| OxideError::InvalidArgument("Expected f32".to_string()))?;
    let b_data = b.storage().as_f32_slice()
        .ok_or_else(|| OxideError::InvalidArgument("Expected f32".to_string()))?;
    let out_data = (*output).storage().as_mut_f32_slice()
        .ok_or_else(|| OxideError::InvalidArgument("Expected f32".to_string()))?;

    let a_dims = a.dims();
    let b_dims = b.dims();
    
    let m = a_dims[a_dims.len() - 2];
    let k = a_dims[a_dims.len() - 1];
    let n = b_dims[b_dims.len() - 1];

    // Simple naive matmul
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a_data[i * k + kk] * b_data[kk * n + j];
            }
            out_data[i * n + j] = sum;
        }
    }

    Ok(())
}

/// Element-wise GPU dispatch
pub fn elementwise_gpu_dispatch<F>(
    a: &Tensor,
    b: Option<&Tensor>,
    output: &mut Tensor,
    op: F,
) -> OxideResult<()>
where
    F: Fn(f32, Option<f32>) -> f32,
{
    let a_data = a.storage().as_f32_slice()
        .ok_or_else(|| OxideError::InvalidArgument("Expected f32".to_string()))?;
    let out_data = (*output).storage().as_mut_f32_slice()
        .ok_or_else(|| OxideError::InvalidArgument("Expected f32".to_string()))?;

    if let Some(b) = b {
        let b_data = b.storage().as_f32_slice()
            .ok_or_else(|| OxideError::InvalidArgument("Expected f32".to_string()))?;
        
        for i in 0..a_data.len() {
            let b_idx = if b_data.len() == a_data.len() { i } else { i % b_data.len() };
            out_data[i] = op(a_data[i], Some(b_data[b_idx]));
        }
    } else {
        for i in 0..a_data.len() {
            out_data[i] = op(a_data[i], None);
        }
    }

    Ok(())
}

/// Normalization GPU dispatch
pub fn rms_norm_gpu_dispatch(
    input: &Tensor,
    weight: &Tensor,
    output: &mut Tensor,
    eps: f32,
) -> OxideResult<()> {
    let input_data = input.storage().as_f32_slice()
        .ok_or_else(|| OxideError::InvalidArgument("Expected f32".to_string()))?;
    let weight_data = weight.storage().as_f32_slice()
        .ok_or_else(|| OxideError::InvalidArgument("Expected f32".to_string()))?;
    let out_data = (*output).storage().as_mut_f32_slice()
        .ok_or_else(|| OxideError::InvalidArgument("Expected f32".to_string()))?;

    let dims = input.dims();
    let num_rows = input.elem_count() / dims[dims.len() - 1];
    let num_cols = dims[dims.len() - 1];

    for row in 0..num_rows {
        let row_offset = row * num_cols;
        
        let mut sum_sq = 0.0f32;
        for i in 0..num_cols {
            let val = input_data[row_offset + i];
            sum_sq += val * val;
        }
        let rms = (sum_sq / num_cols as f32 + eps).sqrt();
        
        for i in 0..num_cols {
            out_data[row_offset + i] = (input_data[row_offset + i] / rms) * weight_data[i];
        }
    }

    Ok(())
}

/// Attention GPU dispatch
pub fn attention_gpu_dispatch(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    output: &mut Tensor,
    scale: f32,
) -> OxideResult<()> {
    let q_data = query.storage().as_f32_slice()
        .ok_or_else(|| OxideError::InvalidArgument("Expected f32".to_string()))?;
    let k_data = key.storage().as_f32_slice()
        .ok_or_else(|| OxideError::InvalidArgument("Expected f32".to_string()))?;
    let v_data = value.storage().as_f32_slice()
        .ok_or_else(|| OxideError::InvalidArgument("Expected f32".to_string()))?;
    let out_data = (*output).storage().as_mut_f32_slice()
        .ok_or_else(|| OxideError::InvalidArgument("Expected f32".to_string()))?;

    let q_shape = query.dims();
    let batch_size = q_shape[0];
    let num_heads = q_shape[1];
    let seq_len = q_shape[2];
    let head_dim = q_shape[q_shape.len() - 1];

    // Naive attention implementation
    for b in 0..batch_size {
        for h in 0..num_heads {
            let q_offset = ((b * num_heads + h) * seq_len * head_dim) as usize;
            let k_offset = ((b * num_heads + h) * seq_len * head_dim) as usize;
            let v_offset = ((b * num_heads + h) * seq_len * head_dim) as usize;
            let out_offset = ((b * num_heads + h) * seq_len * head_dim) as usize;

            for q_pos in 0..seq_len {
                let mut scores = vec![0.0f32; seq_len];
                let mut max_score = f32::NEG_INFINITY;

                for k_pos in 0..seq_len {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_data[q_offset + q_pos * head_dim + d] *
                               k_data[k_offset + k_pos * head_dim + d];
                    }
                    let score = dot * scale;
                    scores[k_pos] = score;
                    max_score = max_score.max(score);
                }

                // Softmax
                let mut sum_exp = 0.0f32;
                for k_pos in 0..seq_len {
                    scores[k_pos] = (scores[k_pos] - max_score).exp();
                    sum_exp += scores[k_pos];
                }
                for k_pos in 0..seq_len {
                    scores[k_pos] /= sum_exp;
                }

                // Weighted sum of values
                for d in 0..head_dim {
                    let mut weighted_sum = 0.0f32;
                    for k_pos in 0..seq_len {
                        weighted_sum += scores[k_pos] * v_data[v_offset + k_pos * head_dim + d];
                    }
                    out_data[out_offset + q_pos * head_dim + d] = weighted_sum;
                }
            }
        }
    }

    Ok(())
}

/// RoPE GPU dispatch
pub fn rope_gpu_dispatch(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    output: &mut Tensor,
) -> OxideResult<()> {
    let x_data = x.storage().as_f32_slice()
        .ok_or_else(|| OxideError::InvalidArgument("Expected f32".to_string()))?;
    let cos_data = cos.storage().as_f32_slice()
        .ok_or_else(|| OxideError::InvalidArgument("Expected f32".to_string()))?;
    let sin_data = sin.storage().as_f32_slice()
        .ok_or_else(|| OxideError::InvalidArgument("Expected f32".to_string()))?;
    let out_data = (*output).storage().as_mut_f32_slice()
        .ok_or_else(|| OxideError::InvalidArgument("Expected f32".to_string()))?;

    let dims = x.dims();
    let seq_len = dims[dims.len() - 2];
    let head_dim = dims[dims.len() - 1];
    let half_dim = head_dim / 2;
    let num_heads = dims[1];

    let num_total = x.elem_count() / head_dim;

    for idx in 0..num_total {
        let pos = idx / num_heads;
        let head = idx % num_heads;
        let base_offset = (pos * num_heads + head) * head_dim;
        let rope_offset = pos * half_dim;

        for d in 0..half_dim {
            let x1 = x_data[base_offset + d];
            let x2 = x_data[base_offset + d + half_dim];
            let c = cos_data[rope_offset + d];
            let s = sin_data[rope_offset + d];

            out_data[base_offset + d] = x1 * c - x2 * s;
            out_data[base_offset + d + half_dim] = x1 * s + x2 * c;
        }
    }

    Ok(())
}

/// Helper trait for GPU-CPU fallback
pub trait GpuFallback: GpuExecutable {
    fn execute_with_fallback(
        &self,
        inputs: &[&Tensor],
        output: &mut Tensor,
    ) -> OxideResult<()> {
        if self.supports_gpu() && inputs.iter().all(|t| t.device().is_cuda()) {
            self.execute_gpu(inputs, output)
        } else {
            // Fall back to CPU
            bail!("GPU not supported, using CPU fallback")
        }
    }
}

/// Macro for defining GPU operations
#[macro_export]
macro_rules! define_gpu_op {
    ($name:ident, $supports:expr, $gpu_impl:expr, $cpu_impl:expr) => {
        pub struct $name;
        
        impl GpuExecutable for $name {
            fn supports_gpu(&self) -> bool {
                $supports
            }
            
            fn execute_gpu(&self, inputs: &[&Tensor], output: &mut Tensor) -> OxideResult<()> {
                $gpu_impl(inputs, output)
            }
        }
    };
}
