use anyhow::{Context, Result};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::data_array::LlamaTokenDataArray;
use std::path::PathBuf;

fn main() -> Result<()> {
    // 1. Initialize logging/tracing
    tracing_subscriber::fmt::init();

    // 2. Initialize the backend
    let backend = LlamaBackend::init()?;

    // 3. Set up model paths and parameters
    let model_path = PathBuf::from("models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf");
    let model_params = LlamaModelParams::default();
    
    println!("Loading model from {}...", model_path.display());

    let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
        .context("Failed to load GGUF model file")?;

    // 4. Create a context
    let ctx_params = LlamaContextParams::default();
    let mut ctx = model
        .new_context(&backend, ctx_params)
        .context("Failed to create llama context")?;

    // 5. Convert your text prompt into model tokens
    let prompt = "The sky is blue because";
    println!("Prompt: \"{}\"\n", prompt);

    let tokens = model
        .str_to_token(prompt, llama_cpp_2::model::AddBos::Always)
        .context("Failed to tokenize prompt")?;

    // 6. Feed the prompt tokens into the context
    let mut batch = llama_cpp_2::llama_batch::LlamaBatch::new(tokens.len(), 1);
    for (i, &token) in tokens.iter().enumerate() {
        batch.add(token, i as i32, &[0], i == tokens.len() - 1)?;
    }
    ctx.decode(&mut batch).context("Failed initial decode")?;

    // 7. Generation loop (Generate up to 30 new tokens)
    let mut current_token_pos = tokens.len() as i32;

    print!("{}", prompt);
    std::io::Write::flush(&mut std::io::stdout())?;

    for _ in 0..30 {
        let candidates_vec: Vec<_> = ctx.candidates_ith(batch.n_tokens() - 1).collect();
        let mut candidates_p = LlamaTokenDataArray::new(candidates_vec, false);
        
        let next_token = candidates_p.sample_token_greedy();

        // Stop generating if the model gives an End-Of-String (EOS) token
        if next_token == model.token_eos() {
            break;
        }

        // FIX: Increased minimum text buffer pre-allocation size from 0 to 32 bytes 
        // to prevent the "Insufficient Buffer Space" crash.
        let token_bytes = model.token_to_piece_bytes(next_token, 32, false, None)?;
        
        // Write the raw bytes directly to the terminal stdout stream safely
        std::io::Write::write_all(&mut std::io::stdout(), &token_bytes)?;
        std::io::Write::flush(&mut std::io::stdout())?;

        // Prepare the new token to feed back into the next loop cycle
        batch.clear();
        batch.add(next_token, current_token_pos, &[0], true)?;
        ctx.decode(&mut batch).context("Failed subsequent decode")?;

        current_token_pos += 1;
    }

    println!("\n\n[Generation Finished]");
    Ok(())
}
