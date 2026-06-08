use crate::core::{Tensor, Shape, Device, DType};
use crate::error::OxideResult;
use crate::models::{LlamaModel, LlamaConfig, MistralModel, MistralConfig};
use crate::cache::KVCache;
use crate::tokenizer::{SimpleTokenizer, InferenceParams, softmax_with_temperature, sample_top_p, sample_top_k};

pub struct InferencePipeline {
    model: Box<dyn LanguageModel>,
    tokenizer: SimpleTokenizer,
    kv_cache: KVCache,
    device: Device,
}

pub trait LanguageModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> OxideResult<Tensor>;
    
    fn vocab_size(&self) -> usize;
    fn hidden_size(&self) -> usize;
    fn num_layers(&self) -> usize;
}

impl LanguageModel for LlamaModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> OxideResult<Tensor> {
        self.forward(input_ids, attention_mask)
    }
    
    fn vocab_size(&self) -> usize {
        self.config().vocab_size
    }
    
    fn hidden_size(&self) -> usize {
        self.config().hidden_size
    }
    
    fn num_layers(&self) -> usize {
        self.config().num_hidden_layers
    }
}

impl LanguageModel for MistralModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> OxideResult<Tensor> {
        self.forward(input_ids, attention_mask)
    }
    
    fn vocab_size(&self) -> usize {
        self.config().vocab_size
    }
    
    fn hidden_size(&self) -> usize {
        self.config().hidden_size
    }
    
    fn num_layers(&self) -> usize {
        self.config().num_hidden_layers
    }
}

impl InferencePipeline {
    pub fn new_llama(
        config: LlamaConfig,
        tokenizer: SimpleTokenizer,
        device: &Device,
    ) -> OxideResult<Self> {
        let model = LlamaModel::new(config, device)?;
        let kv_cache = KVCache::new(
            model.num_layers(),
            1,
            model.config().num_key_value_heads,
            model.config().hidden_size / model.config().num_attention_heads,
            model.config().max_position_embeddings,
            device,
        )?;
        
        Ok(Self {
            model: Box::new(model),
            tokenizer,
            kv_cache,
            device: device.clone(),
        })
    }

    pub fn new_mistral(
        config: MistralConfig,
        tokenizer: SimpleTokenizer,
        device: &Device,
    ) -> OxideResult<Self> {
        let model = MistralModel::new(config, device)?;
        let kv_cache = KVCache::new(
            model.num_layers(),
            1,
            model.config().num_key_value_heads,
            model.config().hidden_size / model.config().num_attention_heads,
            model.config().max_position_embeddings,
            device,
        )?;
        
        Ok(Self {
            model: Box::new(model),
            tokenizer,
            kv_cache,
            device: device.clone(),
        })
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        params: InferenceParams,
    ) -> OxideResult<String> {
        let mut input_ids = self.tokenizer.encode(prompt, true);
        let mut generated_tokens = input_ids.clone();
        let mut generated_text = String::new();

        for _ in 0..params.max_new_tokens {
            let input_tensor = self.prepare_input(&input_ids)?;
            let logits = self.model.forward(&input_tensor, None)?;
            
            let next_token = self.sample_next_token(&logits, &params)?;
            
            if next_token == self.tokenizer.eos_token_id() {
                break;
            }
            
            input_ids.push(next_token);
            generated_tokens.push(next_token);
            
            let token_text = self.tokenizer.decode(&[generated_tokens.len() - 1], false);
            generated_text.push_str(&token_text);
        }

        Ok(self.tokenizer.decode(&generated_tokens, false))
    }

    pub fn generate_streaming(
        &mut self,
        prompt: &str,
        params: InferenceParams,
        mut callback: impl FnMut(&str),
    ) -> OxideResult<String> {
        let mut input_ids = self.tokenizer.encode(prompt, true);
        let mut generated_tokens = input_ids.clone();

        for _ in 0..params.max_new_tokens {
            let input_tensor = self.prepare_input(&input_ids)?;
            let logits = self.model.forward(&input_tensor, None)?;
            
            let next_token = self.sample_next_token(&logits, &params)?;
            
            if next_token == self.tokenizer.eos_token_id() {
                break;
            }
            
            input_ids.push(next_token);
            generated_tokens.push(next_token);
            
            let token_text = self.tokenizer.decode(&[generated_tokens.len() - 1], false);
            callback(&token_text);
        }

        Ok(self.tokenizer.decode(&generated_tokens, false))
    }

    fn prepare_input(&self, input_ids: &[usize]) -> OxideResult<Tensor> {
        let data: Vec<f32> = input_ids.iter().map(|x| *x as f32).collect();
        let tensor = Tensor::from_vec(data, Shape::new(&[1, input_ids.len()]))?;
        tensor.to_device(&self.device)
    }

    fn sample_next_token(
        &self, logits: &Tensor, params: &InferenceParams
    ) -> OxideResult<usize> {
        let logits_vec = logits.to_f32_vec()?;
        let vocab_size = self.model.vocab_size();
        
        let last_logits: Vec<f32> = logits_vec[
            logits_vec.len() - vocab_size..
        ].to_vec();
        
        let probs = softmax_with_temperature(&last_logits, params.temperature);
        
        let probs_with_idx: Vec<(usize, f32)> = probs.iter()
            .enumerate()
            .map(|(i, p)| (i, *p))
            .collect();
        
        let token = if params.top_p < 1.0 {
            sample_top_p(&probs_with_idx, params.top_p)
        } else {
            sample_top_k(&probs_with_idx, params.top_k)
        };
        
        Ok(token)
    }

    pub fn clear_cache(&mut self) {
        self.kv_cache.clear();
    }

    pub fn model_info(&self) -> String {
        format!(
            "Model: vocab_size={}, hidden_size={}, num_layers={}",
            self.model.vocab_size(),
            self.model.hidden_size(),
            self.model.num_layers()
        )
    }
}

use crate::loading::load_safetensors;

pub fn load_model_from_safetensors(
    path: &str,
    config: LlamaConfig,
    tokenizer: SimpleTokenizer,
    device: &Device,
) -> OxideResult<InferencePipeline> {
    let tensors = load_safetensors(path, device)?;
    
    let mut pipeline = InferencePipeline::new_llama(config, tokenizer, device)?;
    
    println!("Loaded {} tensors from {}", tensors.len(), path);
    
    Ok(pipeline)
}

pub fn create_simple_pipeline(device: &Device) -> OxideResult<InferencePipeline> {
    let config = LlamaConfig {
        vocab_size: 100,
        hidden_size: 128,
        intermediate_size: 512,
        num_hidden_layers: 4,
        num_attention_heads: 4,
        num_key_value_heads: 4,
        max_position_embeddings: 512,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
    };
    
    let tokenizer = SimpleTokenizer::from_simple_words(vec![
        "hello", "world", "the", "quick", "brown", "fox",
        "jumps", "over", "lazy", "dog", "cat", "hat",
    ]);
    
    InferencePipeline::new_llama(config, tokenizer, device)
}
