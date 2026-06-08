use crate::core::{Tensor, DType, Device, Shape};
use crate::error::OxideResult;
use crate::ops::{scaled_dot_product_attention, apply_rotary_emb, repeat_kv, silu};
use crate::layers::Linear;
use crate::ops::matmul;

pub struct MultiHeadAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    hidden_dim: usize,
}

impl MultiHeadAttention {
    pub fn new(
        hidden_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        device: &Device,
    ) -> OxideResult<Self> {
        let head_dim = hidden_dim / num_heads;

        let q_proj = Linear::new(hidden_dim, hidden_dim, true, device)?;
        let k_proj = Linear::new(hidden_dim, num_kv_heads * head_dim, true, device)?;
        let v_proj = Linear::new(hidden_dim, num_kv_heads * head_dim, true, device)?;
        let o_proj = Linear::new(hidden_dim, hidden_dim, true, device)?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_dim,
        })
    }

    pub fn from_linear_layers(
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        o_proj: Linear,
        num_heads: usize,
        num_kv_heads: usize,
    ) -> OxideResult<Self> {
        let hidden_dim = q_proj.in_features();
        let head_dim = hidden_dim / num_heads;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_dim,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        cos: Option<&Tensor>,
        sin: Option<&Tensor>,
    ) -> OxideResult<Tensor> {
        let batch_size = hidden_states.dims()[0];
        let seq_len = hidden_states.dims()[1];

        let query = self.q_proj.forward(hidden_states)?;
        let key = self.k_proj.forward(hidden_states)?;
        let value = self.v_proj.forward(hidden_states)?;

        let query = query.reshape(Shape::new(&[batch_size, seq_len, self.num_heads, self.head_dim]))?
            .transpose(1, 2)?;
        let key = key.reshape(Shape::new(&[batch_size, seq_len, self.num_kv_heads, self.head_dim]))?
            .transpose(1, 2)?;
        let value = value.reshape(Shape::new(&[batch_size, seq_len, self.num_kv_heads, self.head_dim]))?
            .transpose(1, 2)?;

        let (query, key) = if let (Some(c), Some(s)) = (cos, sin) {
            let query_rot = apply_rotary_emb(&query, c, s)?;
            let key_rot = apply_rotary_emb(&key, c, s)?;
            (query_rot, key_rot)
        } else {
            (query, key)
        };

        let key = repeat_kv(&key, self.num_heads / self.num_kv_heads)?;
        let value = repeat_kv(&value, self.num_heads / self.num_kv_heads)?;

        let attn_output = scaled_dot_product_attention(&query, &key, &value, attention_mask)?;

        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape(Shape::new(&[batch_size, seq_len, self.hidden_dim]))?;

        self.o_proj.forward(&attn_output)
    }
}

pub struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl MLP {
    pub fn new(hidden_dim: usize, intermediate_dim: usize, device: &Device) -> OxideResult<Self> {
        let gate_proj = Linear::new(hidden_dim, intermediate_dim, true, device)?;
        let up_proj = Linear::new(hidden_dim, intermediate_dim, true, device)?;
        let down_proj = Linear::new(intermediate_dim, hidden_dim, true, device)?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn from_linear_layers(
        gate_proj: Linear,
        up_proj: Linear,
        down_proj: Linear,
    ) -> OxideResult<Self> {
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn forward(&self, x: &Tensor) -> OxideResult<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;

        let gated = silu(&gate)?;
        let activated = crate::ops::mul(&gated, &up)?;

        self.down_proj.forward(&activated)
    }
}
