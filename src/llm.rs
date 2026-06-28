use anyhow::Result;
use llama_cpp_2::{
    llama_backend::LlamaBackend,
    model::{params::LlamaModelParams, LlamaModel},
};

pub struct LLM {
    backend: LlamaBackend,
    model: LlamaModel,
}

impl LLM {
    pub fn new(model_path: &str) -> Result<Self> {
        let backend = LlamaBackend::init()?;

        let model = LlamaModel::load_from_file(
            &backend,
            model_path,
            &LlamaModelParams::default(),
        )?;

        Ok(Self { backend, model })
    }

    pub fn vocab_size(&self) -> i32 {
        self.model.n_vocab()
    }
}