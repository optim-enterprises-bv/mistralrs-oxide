//! Attention CUDA kernels

use cuda_device::{kernel, thread, barrier, warp};

/// Scaled dot-product attention forward pass
#[kernel]
pub fn attention_forward_f32(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    mut output: cuda_device::DisjointSlice<f32>,
    batch_size: u32,
    num_heads: u32,
    seq_len: u32,
    head_dim: u32,
    scale: f32,
) {
    let total = batch_size * num_heads * seq_len;
    let idx = thread::index_1d().get() as u32;
    
    if idx >= total {
        return;
    }

    let batch_idx = idx / (num_heads * seq_len);
    let head_idx = (idx / seq_len) % num_heads;
    let q_pos = idx % seq_len;

    let head_offset = ((batch_idx * num_heads + head_idx) * seq_len * head_dim) as usize;
    let q_offset = head_offset + (q_pos * head_dim) as usize;

    // Compute attention scores for this query position
    let mut max_score = f32::NEG_INFINITY;
    let mut scores: [f32; 1024] = [0.0; 1024];
    
    for k_pos in 0..seq_len {
        let mut dot = 0.0f32;
        for d in 0..head_dim {
            let q_val = query[q_offset + d as usize];
            let k_val = key[head_offset + (k_pos * head_dim + d) as usize];
            dot += q_val * k_val;
        }
        let score = dot * scale;
        scores[k_pos as usize] = score;
        max_score = max_score.max(score);
    }

    // Softmax
    let mut sum_exp = 0.0f32;
    for k_pos in 0..seq_len {
        scores[k_pos as usize] = (scores[k_pos as usize] - max_score).exp();
        sum_exp += scores[k_pos as usize];
    }
    for k_pos in 0..seq_len {
        scores[k_pos as usize] /= sum_exp;
    }

    // Weighted sum of values
    let v_head_offset = ((batch_idx * num_heads + head_idx) * seq_len * head_dim) as usize;
    for d in 0..head_dim {
        let mut weighted_sum = 0.0f32;
        for k_pos in 0..seq_len {
            let v_val = value[v_head_offset + (k_pos * head_dim + d) as usize];
            weighted_sum += scores[k_pos as usize] * v_val;
        }
        let out_idx = head_offset + (q_pos * head_dim + d) as usize;
        if let Some(out) = output.get_mut(cuda_device::ThreadIndex::new(out_idx)) {
            *out = weighted_sum;
        }
    }
}

/// Apply rotary position embeddings (RoPE)
#[kernel]
pub fn apply_rope_f32(
    x: &[f32],
    cos: &[f32],
    sin: &[f32],
    mut output: cuda_device::DisjointSlice<f32>,
    seq_len: u32,
    head_dim: u32,
    num_heads: u32,
) {
    let idx = thread::index_1d().get() as u32;
    let total = seq_len * num_heads * head_dim;
    
    if idx >= total {
        return;
    }

    let pos = idx / (num_heads * head_dim);
    let head = (idx / head_dim) % num_heads;
    let d = idx % head_dim;
    let half_dim = head_dim / 2;

    let base_offset = ((pos * num_heads + head) * head_dim) as usize;
    let rope_offset = (pos * half_dim) as usize;

    if d < half_dim {
        let x1 = x[base_offset + d as usize];
        let x2 = x[base_offset + (d + half_dim) as usize];
        let c = cos[rope_offset + d as usize];
        let s = sin[rope_offset + d as usize];
        
        if let Some(out) = output.get_mut(thread::index_1d()) {
            *out = x1 * c - x2 * s;
        }
    } else {
        let d_2 = d - half_dim;
        let x1 = x[base_offset + d_2 as usize];
        let x2 = x[base_offset + d as usize];
        let c = cos[rope_offset + d_2 as usize];
        let s = sin[rope_offset + d_2 as usize];
        
        if let Some(out) = output.get_mut(thread::index_1d()) {
            *out = x1 * s + x2 * c;
        }
    }
}

/// KV cache update
#[kernel]
pub fn kv_cache_update_f32(
    new_key: &[f32],
    new_value: &[f32],
    mut key_cache: cuda_device::DisjointSlice<f32>,
    mut value_cache: cuda_device::DisjointSlice<f32>,
    batch_size: u32,
    num_heads: u32,
    seq_len: u32,
    head_dim: u32,
    cache_pos: u32,
) {
    let idx = thread::index_1d().get() as u32;
    let total = batch_size * num_heads * seq_len * head_dim;
    
    if idx >= total {
        return;
    }

    let batch = idx / (num_heads * seq_len * head_dim);
    let head = (idx / (seq_len * head_dim)) % num_heads;
    let pos = (idx / head_dim) % seq_len;
    let d = idx % head_dim;

    let cache_offset = ((batch * num_heads + head) * seq_len * head_dim) as usize;
    let cache_idx = cache_offset + (cache_pos * head_dim + d) as usize;
    let new_idx = ((batch * num_heads + head) * seq_len * head_dim + (pos * head_dim + d)) as usize;

    if pos < seq_len {
        if let Some(k_out) = key_cache.get_mut(cuda_device::ThreadIndex::new(cache_idx)) {
            *k_out = new_key[new_idx];
        }
        if let Some(v_out) = value_cache.get_mut(cuda_device::ThreadIndex::new(cache_idx)) {
            *v_out = new_value[new_idx];
        }
    }
}

/// Flash attention (simplified version)
#[kernel]
pub fn flash_attention_f32(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    mut output: cuda_device::DisjointSlice<f32>,
    batch_size: u32,
    num_heads: u32,
    seq_len: u32,
    head_dim: u32,
    scale: f32,
) {
    const BLOCK_SIZE: u32 = 64;
    
    let block_idx = thread::block_idx_1d().x;
    let thread_idx = thread::thread_idx_1d().x;
    
    let q_block_start = block_idx * BLOCK_SIZE;
    let kv_block_start = thread::block_idx_1d().y * BLOCK_SIZE;
    
    if q_block_start >= seq_len || kv_block_start >= seq_len {
        return;
    }

    // Shared memory for tiles
    let mut q_tile: [f32; 64] = [0.0; 64];
    let mut k_tile: [f32; 64] = [0.0; 64];
    let mut v_tile: [f32; 64] = [0.0; 64];
    
    // Load query tile
    let q_pos = q_block_start + thread_idx;
    if q_pos < seq_len {
        for d in 0..head_dim {
            q_tile[d as usize] = query[(q_pos * head_dim + d) as usize] * scale;
        }
    }
    
    barrier::sync();
    
    // Compute attention scores for this block
    if q_pos < seq_len {
        let mut row_max = f32::NEG_INFINITY;
        let mut row_sum = 0.0f32;
        let mut acc: [f32; 64] = [0.0; 64];
        
        // Iterate over KV blocks
        for kv_block in (0..seq_len).step_by(BLOCK_SIZE as usize) {
            // Load KV tile
            barrier::sync();
            
            // Compute scores
            for k_pos in kv_block..(kv_block + BLOCK_SIZE).min(seq_len) {
                let mut score = 0.0f32;
                for d in 0..head_dim {
                    score += q_tile[d as usize] * k_tile[d as usize];
                }
                
                let new_max = row_max.max(score);
                let exp_factor = (row_max - new_max).exp();
                row_sum = row_sum * exp_factor + (score - new_max).exp();
                row_max = new_max;
            }
        }
        
        // Write output
        let out_idx = (q_pos * head_dim) as usize;
        for d in 0..head_dim {
            if let Some(out) = output.get_mut(cuda_device::ThreadIndex::new(out_idx + d as usize)) {
                *out = acc[d as usize] / row_sum;
            }
        }
    }
}

/// Repeat KV heads (for GQA)
#[kernel]
pub fn repeat_kv_f32(
    input: &[f32],
    mut output: cuda_device::DisjointSlice<f32>,
    batch_size: u32,
    num_kv_heads: u32,
    num_q_heads: u32,
    seq_len: u32,
    head_dim: u32,
) {
    let idx = thread::index_1d().get() as u32;
    let repeats = num_q_heads / num_kv_heads;
    let total_out = batch_size * num_q_heads * seq_len * head_dim;
    
    if idx >= total_out {
        return;
    }

    let batch = idx / (num_q_heads * seq_len * head_dim);
    let q_head = (idx / (seq_len * head_dim)) % num_q_heads;
    let pos = (idx / head_dim) % seq_len;
    let d = idx % head_dim;

    let kv_head = q_head / repeats;
    let in_offset = ((batch * num_kv_heads + kv_head) * seq_len * head_dim) as usize;
    let in_idx = in_offset + (pos * head_dim + d) as usize;

    if let Some(out) = output.get_mut(thread::index_1d()) {
        *out = input[in_idx];
    }
}

/// Attention with causal mask
#[kernel]
pub fn causal_attention_f32(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    mut output: cuda_device::DisjointSlice<f32>,
    batch_size: u32,
    num_heads: u32,
    seq_len: u32,
    head_dim: u32,
    scale: f32,
) {
    let total = batch_size * num_heads * seq_len;
    let idx = thread::index_1d().get() as u32;
    
    if idx >= total {
        return;
    }

    let batch_idx = idx / (num_heads * seq_len);
    let head_idx = (idx / seq_len) % num_heads;
    let q_pos = idx % seq_len;

    let head_offset = ((batch_idx * num_heads + head_idx) * seq_len * head_dim) as usize;
    let q_offset = head_offset + (q_pos * head_dim) as usize;

    // Compute attention scores (causal: only attend to previous positions)
    let mut max_score = f32::NEG_INFINITY;
    let mut scores: [f32; 1024] = [0.0; 1024];
    
    for k_pos in 0..=q_pos {
        let mut dot = 0.0f32;
        for d in 0..head_dim {
            let q_val = query[q_offset + d as usize];
            let k_val = key[head_offset + (k_pos * head_dim + d) as usize];
            dot += q_val * k_val;
        }
        let score = dot * scale;
        scores[k_pos as usize] = score;
        max_score = max_score.max(score);
    }

    // Softmax (only over valid positions)
    let mut sum_exp = 0.0f32;
    for k_pos in 0..=q_pos {
        scores[k_pos as usize] = (scores[k_pos as usize] - max_score).exp();
        sum_exp += scores[k_pos as usize];
    }
    for k_pos in 0..=q_pos {
        scores[k_pos as usize] /= sum_exp;
    }

    // Weighted sum
    let v_head_offset = ((batch_idx * num_heads + head_idx) * seq_len * head_dim) as usize;
    for d in 0..head_dim {
        let mut weighted_sum = 0.0f32;
        for k_pos in 0..=q_pos {
            let v_val = value[v_head_offset + (k_pos * head_dim + d) as usize];
            weighted_sum += scores[k_pos as usize] * v_val;
        }
        let out_idx = head_offset + (q_pos * head_dim + d) as usize;
        if let Some(out) = output.get_mut(cuda_device::ThreadIndex::new(out_idx)) {
            *out = weighted_sum;
        }
    }
}
