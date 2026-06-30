use std::path::Path;
use ndarray::Array2;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::execution_providers::CPUExecutionProvider;
use ort::value::TensorRef;
use ort::inputs;

use crate::diarization::types::{DiarizationError, DiarizationConfig, SpeakerEmbedding};

pub struct EmbeddingExtractor {
    session: Session,
}

impl EmbeddingExtractor {
    /// Create a new speaker embedding extractor by loading the ONNX model from the specified path.
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self, DiarizationError> {
        let providers = vec![CPUExecutionProvider::default().build()];
        
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers(providers)?
            .commit_from_file(model_path)?;
            
        Ok(Self { session })
    }

    /// Extract the speaker embedding from the audio samples representing a single segment.
    pub fn extract_embedding(
        &mut self,
        samples: &[f32],
        _config: &DiarizationConfig,
    ) -> Result<SpeakerEmbedding, DiarizationError> {
        // 1. Feature extraction using kaldi-native-fbank.
        // We configure options matching WeSpeaker / 3D-Speaker defaults:
        // - sample_freq: 16000.0
        // - num_bins: 80
        // - frame_shift: 10.0 ms
        // - frame_length: 25.0 ms
        let mut opts = kaldi_native_fbank::FbankOptions::default();
        opts.frame_opts.samp_freq = 16000.0;
        opts.frame_opts.dither = 0.0;
        opts.frame_opts.snip_edges = false;
        opts.mel_opts.num_bins = 80;
        opts.use_energy = false;

        let computer = kaldi_native_fbank::FbankComputer::new(opts)
            .map_err(|e| DiarizationError::FeatureExtraction(e))?;

        let mut fbank = kaldi_native_fbank::OnlineFeature::new(
            kaldi_native_fbank::online::FeatureComputer::Fbank(computer)
        );

        // Feed the waveform at 16kHz
        fbank.accept_waveform(16000.0, samples);
        fbank.input_finished();

        let num_frames = fbank.num_frames_ready();
        let mut feats = Vec::new();
        for i in 0..num_frames {
            if let Some(frame) = fbank.get_frame(i) {
                feats.extend_from_slice(frame);
            }
        }

        // Apply Cepstral Mean Normalization (CMN) per-utterance
        if num_frames > 0 {
            let mut means = vec![0.0f32; 80];
            for i in 0..num_frames {
                for j in 0..80 {
                    means[j] += feats[i * 80 + j];
                }
            }
            for j in 0..80 {
                means[j] /= num_frames as f32;
            }
            for i in 0..num_frames {
                for j in 0..80 {
                    feats[i * 80 + j] -= means[j];
                }
            }
        }

        // Shape of features must be [1, num_frames, 80]
        let feats_tensor = Array2::from_shape_vec((num_frames, 80), feats)?
            .insert_axis(ndarray::Axis(0));
            
        // 2. Feed the log-mel features into the speaker embedding model.
        // WeSpeaker expects input node "feats" with shape [batch_size, time_frames, 80]
        let inputs = inputs![
            "feats" => TensorRef::from_array_view(feats_tensor.view())?
        ];
        
        let outputs = self.session.run(inputs)?;
        
        // Output node is typically "embs", returning a shape [batch_size, 256]
        let embs_val = outputs
            .get("embs")
            .ok_or_else(|| DiarizationError::OutputNotFound("embs".to_string()))?;
            
        let embs_arr = embs_val.try_extract_array::<f32>()?;
        
        // Squeeze batch dimension -> [256]
        let raw_emb = embs_arr.to_owned().remove_axis(ndarray::Axis(0));
        let raw_emb_1d = raw_emb.into_dimensionality::<ndarray::Ix1>()?;
        
        // Perform L2 Normalization (Cosine similarity depends on normalized vectors)
        let mut norm = 0.0f32;
        for val in raw_emb_1d.iter() {
            norm += val * val;
        }
        norm = norm.sqrt();
        
        let normalized_emb = if norm > 1e-6 {
            raw_emb_1d.mapv(|val| val / norm)
        } else {
            raw_emb_1d
        };
        
        Ok(SpeakerEmbedding { data: normalized_emb })
    }
}
