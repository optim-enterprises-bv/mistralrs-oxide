//! Matrix multiplication CUDA kernels

use cuda_device::{kernel, thread, barrier};

/// Naive matrix multiplication: C[M, N] = A[M, K] * B[K, N]
#[kernel]
pub fn matmul_naive_f32(
    a: &[f32],
    b: &[f32],
    mut c: cuda_device::DisjointSlice<f32>,
    m: u32,
    n: u32,
    k: u32,
) {
    let row = thread::index_2d().x as u32;
    let col = thread::index_2d().y as u32;
    
    if row >= m || col >= n {
        return;
    }

    let mut sum = 0.0f32;
    for i in 0..k {
        let a_idx = (row * k + i) as usize;
        let b_idx = (i * n + col) as usize;
        sum += a[a_idx] * b[b_idx];
    }
    
    let c_idx = (row * n + col) as usize;
    if let Some(out) = c.get_mut(thread::index_1d()) {
        *out = sum;
    }
}

/// Tiled matrix multiplication with 1D thread blocks
#[kernel]
pub fn matmul_tiled_1d_f32(
    a: &[f32],
    b: &[f32],
    mut c: cuda_device::DisjointSlice<f32>,
    m: u32,
    n: u32,
    k: u32,
    tile_size: u32,
) {
    let row = thread::index_1d().get() as u32 / tile_size;
    let col = thread::index_1d().get() as u32 % tile_size;
    
    if row >= m || col >= n {
        return;
    }

    let mut sum = 0.0f32;
    
    // Process in tiles
    let num_tiles = (k + tile_size - 1) / tile_size;
    
    for tile in 0..num_tiles {
        let tile_k_start = tile * tile_size;
        let tile_k_end = (tile_k_start + tile_size).min(k);
        
        for kk in tile_k_start..tile_k_end {
            let a_idx = (row * k + kk) as usize;
            let b_idx = (kk * n + col) as usize;
            sum += a[a_idx] * b[b_idx];
        }
    }
    
    let c_idx = (row * n + col) as usize;
    if let Some(out) = c.get_mut(thread::index_1d()) {
        *out = sum;
    }
}

/// Matrix multiplication with shared memory
/// TILE_SIZE must match compile-time constant
#[kernel]
pub fn matmul_shared_f32(
    a: &[f32],
    b: &[f32],
    mut c: cuda_device::DisjointSlice<f32>,
    m: u32,
    n: u32,
    k: u32,
) {
    const TILE_SIZE: u32 = 16;
    
    let block_row = thread::block_idx_2d().x * TILE_SIZE;
    let block_col = thread::block_idx_2d().y * TILE_SIZE;
    let thread_row = thread::thread_idx_2d().x;
    let thread_col = thread::thread_idx_2d().y;
    
    let row = block_row + thread_row;
    let col = block_col + thread_col;
    
    if row >= m || col >= n {
        return;
    }

    // Shared memory tiles
    let mut a_tile: [[f32; 16]; 16] = [[0.0; 16]; 16];
    let mut b_tile: [[f32; 16]; 16] = [[0.0; 16]; 16];
    
    let mut sum = 0.0f32;
    let num_tiles = (k + TILE_SIZE - 1) / TILE_SIZE;
    
    for t in 0..num_tiles {
        // Load tiles into shared memory
        let tiled_k = t * TILE_SIZE;
        
        if tiled_k + thread_col < k {
            a_tile[thread_row as usize][thread_col as usize] = 
                a[(row * k + tiled_k + thread_col) as usize];
        }
        
        if tiled_k + thread_row < k {
            b_tile[thread_row as usize][thread_col as usize] = 
                b[((tiled_k + thread_row) * n + col) as usize];
        }
        
        barrier::sync();
        
        // Compute partial sum
        let tile_k_count = TILE_SIZE.min(k - tiled_k);
        for i in 0..tile_k_count {
            sum += a_tile[thread_row as usize][i as usize] * 
                   b_tile[i as usize][thread_col as usize];
        }
        
        barrier::sync();
    }
    
    let c_idx = (row * n + col) as usize;
    if let Some(out) = c.get_mut(thread::index_1d()) {
        *out = sum;
    }
}

/// Batch matrix multiplication
#[kernel]
pub fn batch_matmul_f32(
    a: &[f32],
    b: &[f32],
    mut c: cuda_device::DisjointSlice<f32>,
    batch_size: u32,
    m: u32,
    n: u32,
    k: u32,
) {
    let batch = thread::index_1d().get() as u32;
    if batch >= batch_size {
        return;
    }
    
    let batch_offset = (batch * m * k) as usize;
    let b_batch_offset = (batch * k * n) as usize;
    let c_batch_offset = (batch * m * n) as usize;
    
    // Compute one batch element
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for i in 0..k {
                let a_idx = batch_offset + (row * k + i) as usize;
                let b_idx = b_batch_offset + (i * n + col) as usize;
                sum += a[a_idx] * b[b_idx];
            }
            
            let c_idx = c_batch_offset + (row * n + col) as usize;
            if let Some(out) = c.get_mut(cuda_device::ThreadIndex::new(c_idx)) {
                *out = sum;
            }
        }
    }
}

/// Matrix-vector multiplication: y = A * x
#[kernel]
pub fn matvec_f32(
    a: &[f32],
    x: &[f32],
    mut y: cuda_device::DisjointSlice<f32>,
    rows: u32,
    cols: u32,
) {
    let row = thread::index_1d().get() as u32;
    if row >= rows {
        return;
    }
    
    let mut sum = 0.0f32;
    for col in 0..cols {
        sum += a[(row * cols + col) as usize] * x[col as usize];
    }
    
    if let Some(out) = y.get_mut(thread::index_1d()) {
        *out = sum;
    }
}

/// Transpose matrix: B = A^T
#[kernel]
pub fn transpose_f32(
    input: &[f32],
    mut output: cuda_device::DisjointSlice<f32>,
    rows: u32,
    cols: u32,
) {
    let row = thread::index_2d().x as u32;
    let col = thread::index_2d().y as u32;
    
    if row >= rows || col >= cols {
        return;
    }
    
    let input_idx = (row * cols + col) as usize;
    let output_idx = (col * rows + row) as usize;
    
    if let Some(out) = output.get_mut(cuda_device::ThreadIndex::new(output_idx)) {
        *out = input[input_idx];
    }
}
