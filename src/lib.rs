pub mod cuda_backend;

pub use cuda_backend::{
    CudaBackend,
    is_cuda_available,
    compute_capability,
    device_count,
};

#[cfg(feature = "cuda")]
pub use cuda_backend::*;

/// GPU feature detection and initialization
pub struct GpuFeatureDetector;

impl GpuFeatureDetector {
    /// Detect available GPU features
    pub fn detect() -> GpuFeatures {
        let cuda_available = is_cuda_available();
        let device_count = if cuda_available {
            device_count()
        } else {
            0
        };
        
        let mut tensor_cores = false;
        let mut max_compute_capability = (0, 0);
        
        for i in 0..device_count {
            if let Some(cc) = compute_capability(i) {
                tensor_cores = tensor_cores || cc.0 >= 7;
                if cc.0 > max_compute_capability.0 ||
                   (cc.0 == max_compute_capability.0 && cc.1 > max_compute_capability.1) {
                    max_compute_capability = cc;
                }
            }
        }
        
        GpuFeatures {
            cuda_available,
            device_count,
            tensor_cores,
            max_compute_capability,
            supports_fp16: tensor_cores,
            supports_bf16: tensor_cores && max_compute_capability.0 >= 8,
            supports_async_copy: max_compute_capability.0 >= 8,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GpuFeatures {
    pub cuda_available: bool,
    pub device_count: usize,
    pub tensor_cores: bool,
    pub max_compute_capability: (i32, i32),
    pub supports_fp16: bool,
    pub supports_bf16: bool,
    pub supports_async_copy: bool,
}

impl GpuFeatures {
    pub fn summary(&self) -> String {
        if !self.cuda_available {
            return "No CUDA devices found".to_string();
        }
        
        format!(
            "CUDA: {} device(s), CC {}.{}, Tensor Cores: {}, FP16: {}, BF16: {}",
            self.device_count,
            self.max_compute_capability.0,
            self.max_compute_capability.1,
            self.tensor_cores,
            self.supports_fp16,
            self.supports_bf16
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_feature_detection() {
        let features = GpuFeatureDetector::detect();
        println!("{}", features.summary());
    }
}
