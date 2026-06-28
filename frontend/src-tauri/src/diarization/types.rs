use ndarray::Array1;
use std::path::PathBuf;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DiarizationSegment {
    pub start: f64,        // start time in seconds
    pub end: f64,          // end time in seconds
    pub speaker_id: usize, // speaker ID (e.g., 0, 1, 2, ...)
}

#[derive(Debug, Clone)]
pub struct SpeakerEmbedding {
    pub data: Array1<f32>, // 256-dimensional embedding vector (L2-normalized)
}

#[derive(Debug, Clone)]
pub struct DiarizationConfig {
    pub segmentation_model_path: PathBuf,
    pub embedding_model_path: PathBuf,
    pub segmentation_window_secs: f64,
    pub segmentation_hop_secs: f64,
    pub clustering_threshold: f32,
    pub sample_rate: u32,
}

impl Default for DiarizationConfig {
    fn default() -> Self {
        Self {
            segmentation_model_path: PathBuf::new(),
            embedding_model_path: PathBuf::new(),
            segmentation_window_secs: 10.0,
            segmentation_hop_secs: 0.5,
            clustering_threshold: 0.8,
            sample_rate: 16000,
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum DiarizationError {
    #[error("ORT error: {0}")]
    Ort(#[from] ort::Error),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("ndarray shape/dimensionality error: {0}")]
    Shape(#[from] ndarray::ShapeError),
    #[error("Model input not found: {0}")]
    InputNotFound(String),
    #[error("Model output not found: {0}")]
    OutputNotFound(String),
    #[error("Feature extraction error: {0}")]
    FeatureExtraction(String),
}
