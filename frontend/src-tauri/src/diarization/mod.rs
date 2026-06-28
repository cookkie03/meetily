pub mod types;
pub mod segmentation;
pub mod embedding;
pub mod clustering;
pub mod commands;

pub use types::{DiarizationSegment, SpeakerEmbedding, DiarizationConfig, DiarizationError};
pub use segmentation::SegmentationEngine;
pub use embedding::EmbeddingExtractor;
pub use clustering::cluster;

use std::path::Path;

pub struct Diarizer {
    segmentation_engine: SegmentationEngine,
    embedding_extractor: EmbeddingExtractor,
}

impl Diarizer {
    /// Initialize a new Diarizer by loading the segmentation and embedding ONNX models.
    pub fn new<P: AsRef<Path>>(
        segmentation_model_path: P,
        embedding_model_path: P,
    ) -> Result<Self, DiarizationError> {
        let segmentation_engine = SegmentationEngine::new(segmentation_model_path)?;
        let embedding_extractor = EmbeddingExtractor::new(embedding_model_path)?;
        
        Ok(Self {
            segmentation_engine,
            embedding_extractor,
        })
    }

    /// Run the end-to-end speaker diarization pipeline on the provided audio samples.
    pub fn diarize(
        &mut self,
        samples: &[f32],
        config: &DiarizationConfig,
    ) -> Result<Vec<DiarizationSegment>, DiarizationError> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }

        // 1. Run speaker segmentation to identify local voice activity segments
        let raw_segments = self.segmentation_engine.segment(samples, config)?;

        // 2. Extract speaker embeddings for each voice segment
        let mut embeddings = Vec::new();
        let mut segments_to_cluster = Vec::new();

        for (start, end, _local_id) in raw_segments {
            let start_sample = (start * config.sample_rate as f64) as usize;
            let end_sample = (end * config.sample_rate as f64) as usize;

            if start_sample < samples.len() && end_sample <= samples.len() && start_sample < end_sample {
                let segment_audio = &samples[start_sample..end_sample];
                
                // Extract embedding from the audio segment
                match self.embedding_extractor.extract_embedding(segment_audio, config) {
                    Ok(emb) => {
                        embeddings.push(emb);
                        segments_to_cluster.push((start, end));
                    }
                    Err(e) => {
                        log::error!("Failed to extract speaker embedding for segment {}-{}: {:?}", start, end, e);
                    }
                }
            }
        }

        // 3. Cluster speaker embeddings to resolve local IDs to global speaker IDs
        let speaker_ids = cluster(&embeddings, config.clustering_threshold);

        // 4. Construct the final diarized segments, sorting and merging contiguous ones of the same speaker
        let mut segments = Vec::new();
        for (idx, (start, end)) in segments_to_cluster.into_iter().enumerate() {
            segments.push(DiarizationSegment {
                start,
                end,
                speaker_id: speaker_ids[idx],
            });
        }

        // Sort by start time (in case they got out of order, though they should be sorted already)
        segments.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap_or(std::cmp::Ordering::Equal));

        let mut merged_segments = Vec::new();
        for seg in segments {
            if merged_segments.is_empty() {
                merged_segments.push(seg);
            } else {
                let last_idx = merged_segments.len() - 1;
                let last = &mut merged_segments[last_idx];
                // If same speaker and contiguous/overlapping (using a 10ms tolerance for rounding)
                if seg.speaker_id == last.speaker_id && seg.start <= last.end + 0.01 {
                    if seg.end > last.end {
                        last.end = seg.end;
                    }
                } else {
                    merged_segments.push(seg);
                }
            }
        }

        Ok(merged_segments)
    }
}
