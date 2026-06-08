//! Benchmark CPU vs GPU operations

use std::time::Instant;
use mistralrs_oxide::{
    core::{Tensor, Shape, Device, DType},
    ops::{matmul, softmax, rms_norm},
    pipeline::create_simple_pipeline,
    GpuFeatureDetector,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Mistralrs-Oxide Benchmark ===\n");
    
    // Detect GPU features
    let features = GpuFeatureDetector::detect();
    println!("GPU Features: {}", features.summary());
    println!();
    
    // Benchmark CPU
    println!("--- CPU Benchmark ---");
    benchmark_cpu();
    
    // Benchmark GPU if available
    if features.cuda_available {
        println!("\n--- GPU Benchmark ---");
        benchmark_gpu()?;
    } else {
        println!("\nGPU not available, skipping GPU benchmarks");
    }
    
    // Model inference benchmark
    println!("\n--- Model Inference Benchmark ---");
    benchmark_inference()?;
    
    Ok(())
}

fn benchmark_cpu() {
    let sizes = vec![128, 256, 512, 1024];
    
    for size in sizes {
        // Matmul benchmark
        let a = Tensor::from_vec(
            vec![0.01f32; size * size],
            Shape::new(&[size, size])
        ).unwrap();
        let b = Tensor::from_vec(
            vec![0.01f32; size * size],
            Shape::new(&[size, size])
        ).unwrap();
        
        let start = Instant::now();
        let _c = matmul(&a, &b).unwrap();
        let elapsed = start.elapsed();
        
        let flops = 2.0 * (size as f64).powi(3);
        let gflops = flops / elapsed.as_secs_f64() / 1e9;
        
        println!(
            "Matmul {}x{}: {:.2?} ({:.2} GFLOPS)",
            size, size, elapsed, gflops
        );
    }
    
    // Softmax benchmark
    let x = Tensor::from_vec(
        vec![1.0f32; 1024 * 1024],
        Shape::new(&[1024, 1024])
    ).unwrap();
    
    let start = Instant::now();
    let _y = softmax(&x, 1
    ).unwrap();
    let elapsed = start.elapsed();
    println!("Softmax 1024x1024: {:.2?}", elapsed);
    
    // RMS norm benchmark
    let x = Tensor::from_vec(
        vec![1.0f32; 4096 * 4096],
        Shape::new(&[4096, 4096])
    ).unwrap();
    let w = Tensor::from_vec(
        vec![1.0f32; 4096],
        Shape::new(&[4096])
    ).unwrap();
    
    let start = Instant::now();
    let _y = rms_norm(&x, &w, 1e-6
    ).unwrap();
    let elapsed = start.elapsed();
    println!("RMSNorm 4096x4096: {:.2?}", elapsed);
}

fn benchmark_gpu() -> Result<(), Box<dyn std::error::Error>> {
    println!("GPU benchmarks require cuda-oxide toolchain");
    println!("Run with: cargo oxide run --example benchmark --features cuda");
    
    // In real implementation:
    // 1. Upload tensors to GPU
    // 2. Launch kernels
    // 3. Compare performance
    // 4. Show speedup ratios
    
    Ok(())
}

fn benchmark_inference() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::default();
    let mut pipeline = create_simple_pipeline(&device)?;
    
    println!("{}", pipeline.model_info());
    
    use mistralrs_oxide::tokenizer::InferenceParams;
    
    let params = InferenceParams {
        max_new_tokens: 20,
        temperature: 1.0,
        top_p: 1.0,
        top_k: 50,
        repetition_penalty: 1.0,
    };
    
    let prompts = vec![
        "hello world",
        "the quick brown fox",
        "once upon a time",
    ];
    
    for prompt in prompts {
        let start = Instant::now();
        let output = pipeline.generate(prompt, params.clone())?;
        let elapsed = start.elapsed();
        
        println!("\nPrompt: \"{}\"", prompt);
        println!("Generated: \"{}\"", output);
        println!("Time: {:.2?}", elapsed);
        println!("Tokens/sec: {:.2}", params.max_new_tokens as f64 / elapsed.as_secs_f64());
    }
    
    Ok(())
}
