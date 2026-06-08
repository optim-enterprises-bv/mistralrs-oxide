//! Normalization CUDA kernels

use cuda_device::{kernel, thread, barrier};

/// RMS normalization
#[kernel]
pub fn rms_norm_f32(
    input: &[f32],
    weight: &[f32],
    mut output: cuda_device::DisjointSlice<f32>,
    num_rows: u32,
    num_cols: u32,
    eps: f32,
) {
    let row = thread::index_1d().get() as u32;
    if row >= num_rows {
        return;
    }

    let row_offset = (row * num_cols) as usize;
    
    // Compute mean square
    let mut sum_sq = 0.0f32;
    for i in 0..num_cols {
        let val = input[row_offset + i as usize];
        sum_sq += val * val;
    }
    let mean_sq = sum_sq / num_cols as f32;
    let rms = (mean_sq + eps).sqrt();
    
    // Normalize and scale
    for i in 0..num_cols {
        let normalized = input[row_offset + i as usize] / rms;
        let out_idx = row_offset + i as usize;
        if let Some(out) = output.get_mut(cuda_device::ThreadIndex::new(out_idx)) {
            *out = normalized * weight[i as usize];
        }
    }
}

/// RMS normalization with fused addition (residual connection)
#[kernel]
pub fn rms_norm_fused_add_f32(
    input: &[f32],
    residual: &[f32],
    weight: &[f32],
    mut output: cuda_device::DisjointSlice<f32>,
    num_rows: u32,
    num_cols: u32,
    eps: f32,
) {
    let row = thread::index_1d().get() as u32;
    if row >= num_rows {
        return;
    }

    let row_offset = (row * num_cols) as usize;
    
    // Add residual first
    let mut sum_sq = 0.0f32;
    for i in 0..num_cols {
        let val = input[row_offset + i as usize] + residual[row_offset + i as usize];
        sum_sq += val * val;
    }
    let mean_sq = sum_sq / num_cols as f32;
    let rms = (mean_sq + eps).sqrt();
    
    // Normalize and scale
    for i in 0..num_cols {
        let val = input[row_offset + i as usize] + residual[row_offset + i as usize];
        let normalized = val / rms;
        let out_idx = row_offset + i as usize;
        if let Some(out) = output.get_mut(cuda_device::ThreadIndex::new(out_idx)) {
            *out = normalized * weight[i as usize];
        }
    }
}

/// Layer normalization
#[kernel]
pub fn layer_norm_f32(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    mut output: cuda_device::DisjointSlice<f32>,
    num_rows: u32,
    num_cols: u32,
    eps: f32,
) {
    let row = thread::index_1d().get() as u32;
    if row >= num_rows {
        return;
    }

    let row_offset = (row * num_cols) as usize;
    
    // Compute mean
    let mut sum = 0.0f32;
    for i in 0..num_cols {
        sum += input[row_offset + i as usize];
    }
    let mean = sum / num_cols as f32;
    
    // Compute variance
    let mut sum_sq = 0.0f32;
    for i in 0..num_cols {
        let diff = input[row_offset + i as usize] - mean;
        sum_sq += diff * diff;
    }
    let var = sum_sq / num_cols as f32;
    let std = (var + eps).sqrt();
    
    // Normalize, scale, and shift
    for i in 0..num_cols {
        let normalized = (input[row_offset + i as usize] - mean) / std;
        let out_idx = row_offset + i as usize;
        if let Some(out) = output.get_mut(cuda_device::ThreadIndex::new(out_idx)) {
            *out = normalized * weight[i as usize] + bias[i as usize];
        }
    }
}

/// Layer normalization with fused addition
#[kernel]
pub fn layer_norm_fused_add_f32(
    input: &[f32],
    residual: &[f32],
    weight: &[f32],
    bias: &[f32],
    mut output: cuda_device::DisjointSlice<f32>,
    num_rows: u32,
    num_cols: u32,
    eps: f32,
) {
    let row = thread::index_1d().get() as u32;
    if row >= num_rows {
        return;
    }

    let row_offset = (row * num_cols) as usize;
    
    // Add residual
    let mut sum = 0.0f32;
    for i in 0..num_cols {
        let val = input[row_offset + i as usize] + residual[row_offset + i as usize];
        sum += val;
    }
    let mean = sum / num_cols as f32;
    
    let mut sum_sq = 0.0f32;
    for i in 0..num_cols {
        let val = input[row_offset + i as usize] + residual[row_offset + i as usize];
        let diff = val - mean;
        sum_sq += diff * diff;
    }
    let var = sum_sq / num_cols as f32;
    let std = (var + eps).sqrt();
    
    for i in 0..num_cols {
        let val = input[row_offset + i as usize] + residual[row_offset + i as usize];
        let normalized = (val - mean) / std;
        let out_idx = row_offset + i as usize;
        if let Some(out) = output.get_mut(cuda_device::ThreadIndex::new(out_idx)) {
            *out = normalized * weight[i as usize] + bias[i as usize];
        }
    }
}

/// Batch normalization (for CNNs, though not typically used in LLMs)
#[kernel]
pub fn batch_norm_f32(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    running_mean: &[f32],
    running_var: &[f32],
    mut output: cuda_device::DisjointSlice<f32>,
    batch_size: u32,
    num_channels: u32,
    spatial_size: u32,
    eps: f32,
) {
    let idx = thread::index_1d().get() as u32;
    let total = batch_size * num_channels * spatial_size;
    
    if idx >= total {
        return;
    }

    let channel = (idx / spatial_size) % num_channels;
    
    let mean = running_mean[channel as usize];
    let var = running_var[channel as usize];
    let std = (var + eps).sqrt();
    
    let normalized = (input[idx as usize] - mean) / std;
    if let Some(out) = output.get_mut(thread::index_1d()) {
        *out = normalized * gamma[channel as usize] + beta[channel as usize];
    }
}

/// Group normalization
#[kernel]
pub fn group_norm_f32(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    mut output: cuda_device::DisjointSlice<f32>,
    batch_size: u32,
    num_channels: u32,
    height: u32,
    width: u32,
    num_groups: u32,
    eps: f32,
) {
    let idx = thread::index_1d().get() as u32;
    let spatial_size = height * width;
    let total = batch_size * num_groups * spatial_size;
    
    if idx >= total {
        return;
    }

    let batch = idx / (num_groups * spatial_size);
    let group = (idx / spatial_size) % num_groups;
    let channels_per_group = num_channels / num_groups;
    
    // Compute mean and variance for this group
    let mut sum = 0.0f32;
    let mut sum_sq = 0.0f32;
    
    for c_in_group in 0..channels_per_group {
        let channel = group * channels_per_group + c_in_group;
        for h in 0..height {
            for w in 0..width {
                let input_idx = ((batch * num_channels + channel) * height + h) * width + w;
                let val = input[input_idx as usize];
                sum += val;
                sum_sq += val * val;
            }
        }
    }
    
    let group_size = (channels_per_group * spatial_size) as f32;
    let mean = sum / group_size;
    let var = sum_sq / group_size - mean * mean;
    let std = (var + eps).sqrt();
    
    // Normalize
    for c_in_group in 0..channels_per_group {
        let channel = group * channels_per_group + c_in_group;
        for h in 0..height {
            for w in 0..width {
                let input_idx = ((batch * num_channels + channel) * height + h) * width + w;
                let normalized = (input[input_idx as usize] - mean) / std;
                if let Some(out) = output.get_mut(cuda_device::ThreadIndex::new(input_idx as usize)) {
                    *out = normalized * weight[channel as usize] + bias[channel as usize];
                }
            }
        }
    }
}

/// Instance normalization
#[kernel]
pub fn instance_norm_f32(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    mut output: cuda_device::DisjointSlice<f32>,
    batch_size: u32,
    num_channels: u32,
    spatial_size: u32,
    eps: f32,
) {
    let idx = thread::index_1d().get() as u32;
    let total = batch_size * num_channels;
    
    if idx >= total {
        return;
    }

    let batch = idx / num_channels;
    let channel = idx % num_channels;
    
    // Compute mean
    let mut sum = 0.0f32;
    for i in 0..spatial_size {
        let input_idx = ((batch * num_channels + channel) * spatial_size + i) as usize;
        sum += input[input_idx];
    }
    let mean = sum / spatial_size as f32;
    
    // Compute variance
    let mut sum_sq = 0.0f32;
    for i in 0..spatial_size {
        let input_idx = ((batch * num_channels + channel) * spatial_size + i) as usize;
        let diff = input[input_idx] - mean;
        sum_sq += diff * diff;
    }
    let var = sum_sq / spatial_size as f32;
    let std = (var + eps).sqrt();
    
    // Normalize
    for i in 0..spatial_size {
        let input_idx = ((batch * num_channels + channel) * spatial_size + i) as usize;
        let normalized = (input[input_idx] - mean) / std;
        if let Some(out) = output.get_mut(cuda_device::ThreadIndex::new(input_idx)) {
            *out = normalized * weight[channel as usize] + bias[channel as usize];
        }
    }
}
