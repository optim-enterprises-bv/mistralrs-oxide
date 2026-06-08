use crate::core::{Tensor, Device};
use crate::error::OxideResult;
use crate::layers::{MultiHeadAttention, MLP, RMSNorm};

pub struct TransformerBlock {
    self_attn: MultiHeadAttention,
    mlp: MLP,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

impl TransformerBlock {
    pub fn new(
        hidden_dim: usize,
        intermediate_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        rms_norm_eps: f32,
        device: &Device,
    ) -> OxideResult<Self> {
        let self_attn = MultiHeadAttention::new(
            hidden_dim,
            num_heads,
            num_kv_heads,
            device,
        )?;
        let mlp = MLP::new(hidden_dim, intermediate_dim, device)?;
        let input_layernorm = RMSNorm::new(hidden_dim, rms_norm_eps, device)?;
        let post_attention_layernorm = RMSNorm::new(hidden_dim, rms_norm_eps, device)?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn from_components(
        self_attn: MultiHeadAttention,
        mlp: MLP,
        input_layernorm: RMSNorm,
        post_attention_layernorm: RMSNorm,
    ) -> Self {
        Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        }
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        cos: Option<&Tensor>,
        sin: Option<&Tensor>,
    ) -> OxideResult<Tensor> {
        let residual = hidden_states;

        let normed = self.input_layernorm.forward(hidden_states)?;
        let attn_output = self.self_attn.forward(&normed, attention_mask, cos, sin)?;
        let hidden_states = crate::ops::add(residual, &attn_output)?;

        let residual = &hidden_states;
        let normed = self.post_attention_layernorm.forward(&hidden_states)?;
        let mlp_output = self.mlp.forward(&normed)?;
        let hidden_states = crate::ops::add(residual, &mlp_output)?;

        Ok(hidden_states)
    }
}
