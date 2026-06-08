use crate::core::{Tensor, Shape, DType, Device};
use crate::error::OxideResult;

pub struct KVCache {
    k_cache: Vec<Tensor>,
    v_cache: Vec<Tensor>,
    current_seq_len: usize,
    max_seq_len: usize,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    batch_size: usize,
}

impl KVCache {
    pub fn new(
        num_layers: usize,
        batch_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        device: &Device,
    ) -> OxideResult<Self> {
        let mut k_cache = Vec::with_capacity(num_layers);
        let mut v_cache = Vec::with_capacity(num_layers);

        for _ in 0..num_layers {
            let k = Tensor::zeros(
                Shape::new(&[batch_size, num_kv_heads, max_seq_len, head_dim]),
                DType::F32,
                device,
            )?;
            let v = Tensor::zeros(
                Shape::new(&[batch_size, num_kv_heads, max_seq_len, head_dim]),
                DType::F32,
                device,
            )?;
            
            k_cache.push(k);
            v_cache.push(v);
        }

        Ok(Self {
            k_cache,
            v_cache,
            current_seq_len: 0,
            max_seq_len,
            num_layers,
            num_kv_heads,
            head_dim,
            batch_size,
        })
    }

    pub fn update(
        &mut self,
        layer_idx: usize,
        key: &Tensor,
        value: &Tensor,
    ) -> OxideResult<()> {
        if layer_idx >= self.num_layers {
            return Err(crate::error::OxideError::InvalidArgument(
                format!("Layer index {} out of range for {} layers", layer_idx, self.num_layers)
            ));
        }

        let key_seq_len = key.dims()[2];
        let end_pos = self.current_seq_len + key_seq_len;

        if end_pos > self.max_seq_len {
            return Err(crate::error::OxideError::InvalidArgument(
                format!("KV cache overflow: {} > {}", end_pos, self.max_seq_len)
            ));
        }

        let mut k_cache = &mut self.k_cache[layer_idx];
        let mut v_cache = &mut self.v_cache[layer_idx];

        let k_slice = k_cache.narrow(2, self.current_seq_len, key_seq_len)?;
        let v_slice = v_cache.narrow(2, self.current_seq_len, key_seq_len)?;

        self.current_seq_len = end_pos;

        Ok(())
    }

    pub fn get(
        &self,
        layer_idx: usize,
        seq_len: Option<usize>,
    ) -> OxideResult<(Option<(&Tensor, &Tensor)>> {
        if layer_idx >= self.num_layers {
            return Err(crate::error::OxideError::InvalidArgument(
                format!("Layer index {} out of range", layer_idx)
            ));
        }

        let actual_seq_len = seq_len.unwrap_or(self.current_seq_len);
        
        if actual_seq_len == 0 {
            return Ok(None);
        }

        let k = &self.k_cache[layer_idx];
        let v = &self.v_cache[layer_idx];

        let k_narrowed = k.narrow(2, 0, actual_seq_len)?;
        let v_narrowed = v.narrow(2, 0, actual_seq_len)?;

        Ok(Some((&k_narrowed, &v_narrowed)))
    }

    pub fn current_seq_len(&self) -> usize {
        self.current_seq_len
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    pub fn clear(&mut self) {
        self.current_seq_len = 0;
    }

    pub fn is_full(&self) -> bool {
        self.current_seq_len >= self.max_seq_len
    }

    pub fn get_cache_size(&self) -> usize {
        let elem_size = 4;
        let cache_per_layer = 2 * self.batch_size * self.num_kv_heads * self.max_seq_len * self.head_dim * elem_size;
        cache_per_layer * self.num_layers
    }
}

pub struct PagedAttentionCache {
    num_blocks: usize,
    block_size: usize,
    num_kv_heads: usize,
    head_dim: usize,
    num_layers: usize,
    cache: Vec<Vec<Tensor>>,
    block_tables: Vec<Vec<usize>>,
    current_seq_lens: Vec<usize>,
}

impl PagedAttentionCache {
    pub fn new(
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        num_layers: usize,
        batch_size: usize,
        device: &Device,
    ) -> OxideResult<Self> {
        let mut cache = Vec::with_capacity(num_layers);
        
        for _ in 0..num_layers {
            let mut layer_cache = Vec::with_capacity(num_blocks);
            for _ in 0..num_blocks {
                let block = Tensor::zeros(
                    Shape::new(&[2, batch_size, num_kv_heads, block_size, head_dim]),
                    DType::F32,
                    device,
                )?;
                layer_cache.push(block);
            }
            cache.push(layer_cache);
        }

        let block_tables: Vec<Vec<usize>> = (0..batch_size)
            .map(|_| Vec::new())
            .collect();
        
        let current_seq_lens = vec![0; batch_size];

        Ok(Self {
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
            num_layers,
            cache,
            block_tables,
            current_seq_lens,
        })
    }

    pub fn allocate_blocks(&mut self, batch_idx: usize, num_blocks: usize) -> OxideResult<()> {
        Ok(())
    }

    pub fn get_num_allocated_blocks(&self) -> usize {
        self.block_tables.iter().map(|bt| bt.len()).sum()
    }

    pub fn clear_batch(&mut self, batch_idx: usize) {
        self.block_tables[batch_idx].clear();
        self.current_seq_lens[batch_idx] = 0;
    }
}

pub struct PrefixCache {
    cache: std::collections::HashMap<String, (Tensor, Tensor, usize)>,
}

impl PrefixCache {
    pub fn new() -> Self {
        Self {
            cache: std::collections::HashMap::new(),
        }
    }

    pub fn get(&self, prefix: &str) -> Option<(&Tensor, &Tensor, usize)> {
        self.cache.get(prefix).map(|(k, v, len)| (k, v, *len))
    }

    pub fn insert(&mut self,
        prefix: String,
        key: Tensor,
        value: Tensor,
        seq_len: usize,
    ) {
        self.cache.insert(prefix, (key, value, seq_len));
    }

    pub fn clear(&mut self) {
        self.cache.clear();
    }
}
