use crate::core::{Tensor, Device, Shape, DType};
use crate::error::OxideResult;
use crate::layers::{Embedding, RMSNorm, TransformerBlock};

pub struct MistralConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub sliding_window: Option<usize>,
}

impl Default for MistralConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            max_position_embeddings: 8192,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            sliding_window: Some(4096),
        }
    }
}

pub struct MistralModel {
    embed_tokens: Embedding,
    layers: Vec<TransformerBlock>,
    norm: RMSNorm,
    lm_head: crate::layers::Linear,
    config: MistralConfig,
    cos_cache: Tensor,
    sin_cache: Tensor,
}

impl MistralModel {
    pub fn new(config: MistralConfig, device: &Device) -> OxideResult<Self> {
        let embed_tokens = Embedding::new(
            config.vocab_size,
            config.hidden_size,
            device,
        )?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(TransformerBlock::new(
                config.hidden_size,
                config.intermediate_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.rms_norm_eps,
                device,
            )?);
        }

        let norm = RMSNorm::new(config.hidden_size, config.rms_norm_eps, device)?;
        let lm_head = crate::layers::Linear::new(
            config.hidden_size,
            config.vocab_size,
            false,
            device,
        )?;

        let (cos_cache, sin_cache) = Self::precompute_rope_cache(
            config.max_position_embeddings,
            config.hidden_size / config.num_attention_heads,
            config.rope_theta,
            device,
        )?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            config,
            cos_cache,
            sin_cache,
        })
    }

    fn precompute_rope_cache(
        max_seq_len: usize,
        head_dim: usize,
        theta: f32,
        device: &Device,
    ) -> OxideResult<(Tensor, Tensor)> {
        let half_dim = head_dim / 2;
        
        let mut cos_vals = vec![0.0f32; max_seq_len * half_dim];
        let mut sin_vals = vec![0.0f32; max_seq_len * half_dim];

        for pos in 0..max_seq_len {
            for i in 0..half_dim {
                let freq = theta.powf(-2.0 * i as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                cos_vals[pos * half_dim + i] = angle.cos();
                sin_vals[pos * half_dim + i] = angle.sin();
            }
        }

        let cos = Tensor::from_vec(cos_vals, Shape::new(&[max_seq_len, half_dim]))?;
        let sin = Tensor::from_vec(sin_vals, Shape::new(&[max_seq_len, half_dim]))?;

        Ok((cos, sin))
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> OxideResult<Tensor> {
        let seq_len = input_ids.dims()[1];
        
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        let cos = self.cos_cache.narrow(0, 0, seq_len)?;
        let sin = self.sin_cache.narrow(0, 0, seq_len)?;

        for layer in &self.layers {
            hidden_states = layer.forward(
                &hidden_states,
                attention_mask,
                Some(&cos),
                Some(&sin),
            )?;
        }

        hidden_states = self.norm.forward(&hidden_states)?;
        let logits = self.lm_head.forward(&hidden_states)?;

        Ok(logits)
    }

    pub fn config(&self) -> &MistralConfig {
        &self.config
    }
}
