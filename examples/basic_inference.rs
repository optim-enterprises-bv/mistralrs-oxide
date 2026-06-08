use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Mistralrs-Oxide Inference Example ===\n");
    
    let device = mistralrs_oxide::Device::default();
    println!("Using device: {:?}", device);
    
    let mut pipeline = mistralrs_oxide::pipeline::create_simple_pipeline(&device
    )?;
    
    println!("{}", pipeline.model_info());
    println!("Model initialized successfully!\n");
    
    let prompt = "hello world";
    println!("Prompt: \"{}\"", prompt);
    
    let params = mistralrs_oxide::tokenizer::InferenceParams {
        max_new_tokens: 10,
        temperature: 1.0,
        top_p: 1.0,
        top_k: 50,
        repetition_penalty: 1.0,
    };
    
    println!("Generating...\n");
    
    let output = pipeline.generate(prompt, params)?;
    
    println!("Generated: \"{}\"", output);
    
    Ok(())
}
