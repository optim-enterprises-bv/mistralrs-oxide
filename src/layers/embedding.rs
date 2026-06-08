use crate::core::{Tensor, DType, Device, Shape};
use crate::error::OxideResult;
use crate::ops::embedding as embedding_op;

pub struct Embedding {
    weight: Tensor,
    num_embeddings: usize,
    embedding_dim: usize,
}

impl Embedding {
    pub fn new(num_embeddings: usize, embedding_dim: usize, device: &Device) -> OxideResult<Self> {
        let weight = Tensor::from_vec(
            vec![0.001f32; num_embeddings * embedding_dim],
            Shape::new(&[num_embeddings, embedding_dim])
        )?;

        Ok(Self {
            weight,
            num_embeddings,
            embedding_dim,
        })
    }

    pub fn from_tensor(weight: Tensor) -> OxideResult<Self> {
        let dims = weight.dims();
        if dims.len() != 2 {
            return Err(crate::error::OxideError::InvalidArgument(
                "Embedding weight must be 2D".to_string()
            ));
        }
        
        Ok(Self {
            weight,
            num_embeddings: dims[0],
            embedding_dim: dims[1],
        })
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn forward(&self, indices: &Tensor) -> OxideResult<Tensor> {
        embedding_op(indices, &self.weight)
    }

    pub fn num_embeddings(&self) -> usize {
        self.num_embeddings
    }

    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}
