use std::collections::HashMap;
use crate::error::{OxideError, OxideResult};

pub struct SimpleTokenizer {
    vocab: HashMap<String, usize>,
    reverse_vocab: HashMap<usize, String>,
    bos_token: usize,
    eos_token: usize,
    pad_token: usize,
    unk_token: usize,
}

impl SimpleTokenizer {
    pub fn new(vocab_size: usize) -> Self {
        let mut vocab = HashMap::new();
        let mut reverse_vocab = HashMap::new();

        vocab.insert("<s>".to_string(), 0);
        vocab.insert("</s>".to_string(), 1);
        vocab.insert("<pad>".to_string(), 2);
        vocab.insert("<unk>".to_string(), 3);

        reverse_vocab.insert(0, "<s>".to_string());
        reverse_vocab.insert(1, "</s>".to_string());
        reverse_vocab.insert(2, "<pad>".to_string());
        reverse_vocab.insert(3, "<unk>".to_string());

        for i in 4..vocab_size {
            let token = format!("token_{}", i);
            vocab.insert(token.clone(), i);
            reverse_vocab.insert(i, token);
        }

        Self {
            vocab,
            reverse_vocab,
            bos_token: 0,
            eos_token: 1,
            pad_token: 2,
            unk_token: 3,
        }
    }

    pub fn from_simple_words(words: Vec<&str>) -> Self {
        let mut vocab = HashMap::new();
        let mut reverse_vocab = HashMap::new();

        vocab.insert("<s>".to_string(), 0);
        vocab.insert("</s>".to_string(), 1);
        vocab.insert("<pad>".to_string(), 2);
        vocab.insert("<unk>".to_string(), 3);
        vocab.insert("<space>".to_string(), 4);

        reverse_vocab.insert(0, "<s>".to_string());
        reverse_vocab.insert(1, "</s>".to_string());
        reverse_vocab.insert(2, "<pad>".to_string());
        reverse_vocab.insert(3, "<unk>".to_string());
        reverse_vocab.insert(4, " ".to_string());

        let mut idx = 5;
        for word in words {
            vocab.insert(word.to_string(), idx);
            reverse_vocab.insert(idx, word.to_string());
            idx += 1;
        }

        Self {
            vocab,
            reverse_vocab,
            bos_token: 0,
            eos_token: 1,
            pad_token: 2,
            unk_token: 3,
        }
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool
    ) -> Vec<usize> {
        let mut tokens = Vec::new();

        if add_special_tokens {
            tokens.push(self.bos_token);
        }

        let words: Vec<&str> = text.split_whitespace().collect();
        for word in words {
            if let Some(&idx) = self.vocab.get(word) {
                tokens.push(idx);
            } else {
                tokens.push(self.unk_token);
            }
            tokens.push(self.vocab["<space>"]);
        }

        if add_special_tokens {
            tokens.push(self.eos_token);
        }

        tokens
    }

    pub fn decode(&self, tokens: &[usize], skip_special_tokens: bool
    ) -> String {
        let mut result = String::new();

        for &token in tokens {
            if skip_special_tokens && (token == self.bos_token ||
                token == self.eos_token ||
                token == self.pad_token) {
                continue;
            }

            if let Some(token_str) = self.reverse_vocab.get(&token) {
                if token_str == "<space>" {
                    result.push(' ');
                } else if !token_str.starts_with("token_") {
                    result.push_str(token_str);
                } else {
                    result.push('?');
                }
            } else {
                result.push('?');
            }
        }

        result
    }

    pub fn bos_token_id(&self) -> usize {
        self.bos_token
    }

    pub fn eos_token_id(&self) -> usize {
        self.eos_token
    }

    pub fn pad_token_id(&self) -> usize {
        self.pad_token
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

pub struct InferenceParams {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
}

impl Default for InferenceParams {
    fn default() -> Self {
        Self {
            max_new_tokens: 100,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 50,
            repetition_penalty: 1.0,
        }
    }
}

pub fn softmax_with_temperature(logits: &Vec<f32>, temperature: f32) -> Vec<f32> {
    let scaled: Vec<f32> = logits.iter()
        .map(|x| x / temperature)
        .collect();

    let max_logit = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let exp_sum: f32 = scaled.iter()
        .map(|x| (x - max_logit).exp())
        .sum();

    scaled.iter()
        .map(|x| (x - max_logit).exp() / exp_sum)
        .collect()
}

pub fn sample_top_p(probs: &[(usize, f32)], top_p: f32) -> usize {
    let mut sorted = probs.to_vec();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut cumsum = 0.0f32;
    let mut n_keep = sorted.len();

    for (i, (_, prob)) in sorted.iter().enumerate() {
        cumsum += prob;
        if cumsum >= top_p {
            n_keep = i + 1;
            break;
        }
    }

    let top_probs = &sorted[..n_keep];
    let total: f32 = top_probs.iter().map(|(_, p)| p).sum();

    let r = fastrand::f32() * total;
    let mut cum = 0.0f32;

    for (idx, prob) in top_probs {
        cum += prob;
        if cum >= r {
            return *idx;
        }
    }

    top_probs.last().map(|(idx, _)| *idx).unwrap_or(0)
}

pub fn sample_top_k(probs: &[(usize, f32)], top_k: usize) -> usize {
    let k = top_k.min(probs.len());
    let mut sorted = probs.to_vec();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let top_k_probs = &sorted[..k];
    let total: f32 = top_k_probs.iter().map(|(_, p)| p).sum();

    let r = fastrand::f32() * total;
    let mut cum = 0.0f32;

    for (idx, prob) in top_k_probs {
        cum += prob;
        if cum >= r {
            return *idx;
        }
    }

    top_k_probs.last().map(|(idx, _)| *idx).unwrap_or(0)
}
