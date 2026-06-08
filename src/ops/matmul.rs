use crate::core::{Device, DType, Shape, Tensor, Storage, StorageData, CpuStorage, Layout};
use crate::error::{OxideError, OxideResult, bail};

pub fn matmul(a: &Tensor, b: &Tensor) -> OxideResult<Tensor> {
    let a_dims = a.dims();
    let b_dims = b.dims();
    
    if a_dims.len() < 2 || b_dims.len() < 2 {
        bail!("matmul requires 2D or higher dimensional tensors");
    }

    let a_rows = a_dims[a_dims.len() - 2];
    let a_cols = a_dims[a_dims.len() - 1];
    let b_rows = b_dims[b_dims.len() - 2];
    let b_cols = b_dims[b_dims.len() - 1];

    if a_cols != b_rows {
        bail!("matmul: dimension mismatch {:?} @ {:?} - {} != {}",
            a_dims, b_dims, a_cols, b_rows);
    }

    if !a.device().same_device(&b.device()) {
        bail!("matmul requires tensors on same device");
    }

    let batch_a: Vec<usize> = a_dims[..a_dims.len()-2].to_vec();
    let batch_b: Vec<usize> = b_dims[..b_dims.len()-2].to_vec();
    let batch = broadcast_batch(&batch_a, &batch_b)?;

    let mut out_shape = batch.clone();
    out_shape.push(a_rows);
    out_shape.push(b_cols);

    let output = Tensor::zeros(Shape::new(&out_shape), a.dtype(), &a.device())?;

    if a.device().is_cpu() {
        let a_data = a.storage().as_f32_slice().ok_or_else(|| 
            OxideError::InvalidArgument("Expected f32 tensor".to_string()))?;
        let b_data = b.storage().as_f32_slice().ok_or_else(|| 
            OxideError::InvalidArgument("Expected f32 tensor".to_string()))?;
        
        let mut out_storage = (*output.storage()).clone();
        let out_data = out_storage.as_mut_f32_slice().ok_or_else(|| 
            OxideError::InvalidArgument("Expected f32 tensor".to_string()))?;

        let batch_size = batch.iter().product::<usize>().max(1);
        
        for batch_idx in 0..batch_size {
            let a_offset = batch_idx * a_rows * a_cols;
            let b_offset = batch_idx * b_rows * b_cols;
            let out_offset = batch_idx * a_rows * b_cols;
            
            for i in 0..a_rows {
                for j in 0..b_cols {
                    let mut sum = 0.0f32;
                    for k in 0..a_cols {
                        sum += a_data[a_offset + i * a_cols + k] * 
                               b_data[b_offset + k * b_cols + j];
                    }
                    out_data[out_offset + i * b_cols + j] = sum;
                }
            }
        }
    }

    Ok(output)
}

pub fn matmul_with_bias(a: &Tensor, b: &Tensor, bias: &Tensor) -> OxideResult<Tensor> {
    let c = matmul(a, b)?;
    add(&c, bias)
}

pub fn add(a: &Tensor, b: &Tensor) -> OxideResult<Tensor> {
    let a_device = a.device();
    let b_device = b.device();
    
    let (a, b) = if !a_device.same_device(&b_device) {
        let target = if a_device.is_cuda() { &a_device } else { &b_device };
        (a.to_device(target)?, b.to_device(target)?)
    } else {
        (a.clone(), b.clone())
    };

    let out_shape = broadcast_shape(a.dims(), b.dims())?;
    let output = Tensor::zeros(Shape::new(&out_shape), a.dtype(), &a.device())?;

    if a.device().is_cpu() {
        elementwise_op_cpu(&a, &b, &mut (*output.storage()).clone(),
            |x, y| x + y
        )?;
    }

    Ok(output)
}

pub fn mul(a: &Tensor, b: &Tensor) -> OxideResult<Tensor> {
    let out_shape = broadcast_shape(a.dims(), b.dims())?;
    let output = Tensor::zeros(Shape::new(&out_shape), a.dtype(), &a.device())?;

    if a.device().is_cpu() {
        elementwise_op_cpu(&a, &b, &mut (*output.storage()).clone(),
            |x, y| x * y
        )?;
    }

    Ok(output)
}

pub fn sub(a: &Tensor, b: &Tensor) -> OxideResult<Tensor> {
    let out_shape = broadcast_shape(a.dims(), b.dims())?;
    let output = Tensor::zeros(Shape::new(&out_shape), a.dtype(), &a.device())?;

    if a.device().is_cpu() {
        elementwise_op_cpu(&a, &b, &mut (*output.storage()).clone(),
            |x, y| x - y
        )?;
    }

    Ok(output)
}

pub fn div(a: &Tensor, b: &Tensor) -> OxideResult<Tensor> {
    let out_shape = broadcast_shape(a.dims(), b.dims())?;
    let output = Tensor::zeros(Shape::new(&out_shape), a.dtype(), &a.device())?;

    if a.device().is_cpu() {
        elementwise_op_cpu(&a, &b, &mut (*output.storage()).clone(),
            |x, y| x / y
        )?;
    }

    Ok(output)
}

fn elementwise_op_cpu<F>(
    a: &Tensor,
    b: &Tensor,
    out: &mut Storage,
    op: F
) -> OxideResult<()>
where
    F: Fn(f32, f32) -> f32,
{
    let a_data = a.storage().as_f32_slice().ok_or_else(|| 
        OxideError::InvalidArgument("Expected f32 tensor".to_string()))?;
    let b_data = b.storage().as_f32_slice().ok_or_else(|| 
        OxideError::InvalidArgument("Expected f32 tensor".to_string()))?;
    let out_data = out.as_mut_f32_slice().ok_or_else(|| 
        OxideError::InvalidArgument("Expected f32 tensor".to_string()))?;

    let count = out.layout.num_elements();
    for i in 0..count {
        out_data[i] = op(a_data[i % a_data.len()], b_data[i % b_data.len()]);
    }

    Ok(())
}

pub fn broadcast_shape(a: &[usize], b: &[usize]) -> OxideResult<Vec<usize>> {
    let max_rank = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_rank);

    for i in 0..max_rank {
        let a_dim = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let b_dim = if i < b.len() { b[b.len() - 1 - i] } else { 1 };

        if a_dim != b_dim && a_dim != 1 && b_dim != 1 {
            bail!("broadcast: shape mismatch {:?} and {:?}", a, b);
        }

        result.push(a_dim.max(b_dim));
    }

    result.reverse();
    Ok(result)
}

fn broadcast_batch(a: &[usize], b: &[usize]) -> OxideResult<Vec<usize>> {
    broadcast_shape(a, b)
}
