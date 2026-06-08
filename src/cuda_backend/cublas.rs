//! Real cuBLAS integration using cudarc

#![cfg(feature = "cuda")]

use std::sync::Arc;
use cudarc::driver::{CudaDevice, CudaSlice};
use cudarc::cublas::{Cublas, GemmConfig as CublasGemmConfig};
use crate::error::{OxideError, OxideResult};

/// cuBLAS context wrapper
pub struct CublasContext {
    cublas: Cublas,
}

impl CublasContext {
    /// Create new cuBLAS context
    pub fn new(device: &Arc<CudaDevice>) -> Option<Self> {
        Cublas::new(device).ok().map(|cublas| Self { cublas })
    }

    /// GEMM: C = alpha * A * B + beta * C
    pub fn gemm(
        &self,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        beta: f32,
    ) -> OxideResult<()> {
        let config = CublasGemmConfig {
            alpha,
            beta,
            m: m as i32,
            n: n as i32,
            k: k as i32,
            lda: k as i32,  // leading dimension of A
            ldb: n as i32,  // leading dimension of B
            ldc: n as i32,  // leading dimension of C
            transa: false,
            transb: false,
        };

        self.cublas.gemm(config, a, b, c)
            .map_err(|e| OxideError::CudaError(format!("cuBLAS GEMM failed: {}", e)))
    }

    /// GEMM with FP16 (if available)
    pub fn gemm_f16(
        &self,
        a: &CudaSlice<f16>,
        b: &CudaSlice<f16>,
        c: &mut CudaSlice<f16>,
        m: usize,
        n: usize,
        k: usize,
        alpha: f16,
        beta: f16,
    ) -> OxideResult<()> {
        // cudarc may not expose HGEMM directly
        // Would need to use cublasHgemm or cublasGemmEx
        Err(OxideError::UnsupportedOperation(
            "FP16 GEMM not yet supported".to_string()
        ))
    }

    /// Batched GEMM
    pub fn gemm_batched(
        &self,
        a: &[&CudaSlice<f32>],
        b: &[&CudaSlice<f32>],
        c: &mut [&mut CudaSlice<f32>],
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        beta: f32,
    ) -> OxideResult<()> {
        // Batch size
        let batch_size = a.len();
        assert_eq!(b.len(), batch_size);
        assert_eq!(c.len(), batch_size);

        for i in 0..batch_size {
            self.gemm(a[i], b[i], c[i], m, n, k, alpha, beta)?;
        }

        Ok(())
    }
}

/// GEMM configuration
#[derive(Debug, Clone)]
pub struct GemmConfig {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub alpha: f32,
    pub beta: f32,
    pub trans_a: bool,
    pub trans_b: bool,
}

impl GemmConfig {
    pub fn new(m: usize, n: usize, k: usize) -> Self {
        Self {
            m,
            n,
            k,
            alpha: 1.0,
            beta: 0.0,
            trans_a: false,
            trans_b: false,
        }
    }

    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn with_beta(mut self, beta: f32) -> Self {
        self.beta = beta;
        self
    }

    pub fn with_transpose(mut self, trans_a: bool, trans_b: bool) -> Self {
        self.trans_a = trans_a;
        self.trans_b = trans_b;
        self
    }
}

/// cuBLAS wrapper with automatic algorithm selection
pub struct CublasWrapper {
    cublas: CublasContext,
    use_tensor_cores: bool,
}

impl CublasWrapper {
    pub fn new(cublas: CublasContext) -> Self {
        Self {
            cublas,
            use_tensor_cores: true,
        }
    }

    pub fn set_tensor_cores(&mut self, enabled: bool) {
        self.use_tensor_cores = enabled;
    }

    /// Smart GEMM: automatically choose best algorithm
    pub fn gemm_smart(
        &self,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> OxideResult<()> {
        // For now, just use standard GEMM
        // In production, would check sizes and choose best algorithm
        self.cublas.gemm(a, b, c, m, n, k, 1.0, 0.0)
    }
}

use half::f16;
