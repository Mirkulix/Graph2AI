use std::path::{Path, PathBuf};

use qlang_runtime::hybrid_router::{
    default_hybrid_router_dataset, HybridRouterDataset, HybridRouterSpecialists,
};

pub fn hybrid_router_dataset_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("datasets")
        .join("hybrid_router_training.json")
}

pub fn hybrid_router_model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("artifacts")
        .join("hybrid_router_specialists.json")
}

pub fn load_dataset(path: &Path) -> Result<HybridRouterDataset, String> {
    HybridRouterDataset::load_from_path(path)
}

pub fn load_dataset_with_fallback(path: &Path) -> HybridRouterDataset {
    match load_dataset(path) {
        Ok(dataset) => dataset,
        Err(err) => {
            eprintln!("[hybrid-router] dataset fallback: {err}");
            default_hybrid_router_dataset()
        }
    }
}

pub fn load_specialists(path: &Path) -> Result<HybridRouterSpecialists, String> {
    HybridRouterSpecialists::load_from_path(path)
}
