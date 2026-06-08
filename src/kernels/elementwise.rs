//! Element-wise CUDA kernels

use cuda_device::{kernel, thread};

/// Element-wise addition kernel
#[kernel]
pub fn add_f32(
    a: &[f32],
    b: &[f32],
    mut c: cuda_device::DisjointSlice<f32>,
    n: u32,
) {
    let idx = thread::index_1d().get() as u32;
    if idx < n {
        if let Some(out) = c.get_mut(thread::index_1d()) {
            *out = a[idx as usize] + b[idx as usize];
        }
    }
}

/// Element-wise subtraction kernel
#[kernel]
pub fn sub_f32(
    a: &[f32],
    b: &[f32],
    mut c: cuda_device::DisjointSlice<f32>,
    n: u32,
) {
    let idx = thread::index_1d().get() as u32;
    if idx < n {
        if let Some(out) = c.get_mut(thread::index_1d()) {
            *out = a[idx as usize] - b[idx as usize];
        }
    }
}

/// Element-wise multiplication kernel
#[kernel]
pub fn mul_f32(
    a: &[f32],
    b: &[f32],
    mut c: cuda_device::DisjointSlice<f32>,
    n: u32,
) {
    let idx = thread::index_1d().get() as u32;
    if idx < n {
        if let Some(out) = c.get_mut(thread::index_1d()) {
            *out = a[idx as usize] * b[idx as usize];
        }
    }
}

/// Element-wise division kernel
#[kernel]
pub fn div_f32(
    a: &[f32],
    b: &[f32],
    mut c: cuda_device::DisjointSlice<f32>,
    n: u32,
) {
    let idx = thread::index_1d().get() as u32;
    if idx < n {
        if let Some(out) = c.get_mut(thread::index_1d()) {
            let divisor = b[idx as usize];
            *out = if divisor != 0.0 { a[idx as usize] / divisor } else { 0.0 };
        }
    }
}

/// Scale kernel: out[i] = in[i] * scale
#[kernel]
pub fn scale_f32(
    input: &[f32],
    mut output: cuda_device::DisjointSlice<f32>,
    scale: f32,
    n: u32,
) {
    let idx = thread::index_1d().get() as u32;
    if idx < n {
        if let Some(out) = output.get_mut(thread::index_1d()) {
            *out = input[idx as usize] * scale;
        }
    }
}

/// Broadcast binary operation
#[kernel]
pub fn broadcast_add_f32(
    a: &[f32],
    b: &[f32],
    mut c: cuda_device::DisjointSlice<f32>,
    a_shape: [u32; 8],
    b_shape: [u32; 8],
    out_shape: [u32; 8],
    rank: u32,
) {
    // Complex broadcast logic - simplified version
    let idx = thread::index_1d().get() as u32;
    let out_elems: u32 = out_shape[..rank as usize].iter().product();
    
    if idx >= out_elems {
        return;
    }
    
    // Simplified: assume matching shapes for now
    if let Some(out) = c.get_mut(thread::index_1d()) {
        let a_idx = idx as usize % a_shape.iter().product::<u32>() as usize;
        let b_idx = idx as usize % b_shape.iter().product::<u32>() as usize;
        *out = a[a_idx] + b[b_idx];
    }
}
