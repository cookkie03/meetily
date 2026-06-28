use std::path::Path;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::execution_providers::CPUExecutionProvider;
use ort::value::TensorRef;
use ort::inputs;
use ndarray::Array3;

use crate::diarization::types::{DiarizationError, DiarizationConfig};

pub struct SegmentationEngine {
    session: Session,
}

impl SegmentationEngine {
    /// Create a new instance of the segmentation engine by loading the ONNX model from the specified path.
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self, DiarizationError> {
        let providers = vec![CPUExecutionProvider::default().build()];
        
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers(providers)?
            .commit_from_file(model_path)?;
            
        Ok(Self { session })
    }

    /// Process the audio samples and segment them.
    /// In this phase (1a), we lay the foundations of the ORT session run and parameters.
    pub fn segment(
        &mut self,
        samples: &[f32],
        config: &DiarizationConfig,
    ) -> Result<Vec<(f64, f64, usize)>, DiarizationError> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }

        let sample_rate = config.sample_rate as f64;
        let window_samples = (config.segmentation_window_secs * sample_rate) as usize;
        let hop_samples = (config.segmentation_hop_secs * sample_rate) as usize;

        let total_duration = samples.len() as f64 / sample_rate;

        // Build list of window start sample indices
        let mut w_starts = Vec::new();
        let mut w_idx = 0;
        while w_idx < samples.len() {
            w_starts.push(w_idx);
            w_idx += hop_samples;
        }

        let mut num_frames_per_window = None;
        let mut delta_t = None;

        let mut global_activity: Vec<[f32; 3]> = Vec::new();
        let mut global_weight: Vec<f32> = Vec::new();

        for &w_start in &w_starts {
            let mut window_audio = vec![0.0f32; window_samples];
            let w_end = w_start + window_samples;
            if w_end <= samples.len() {
                window_audio.copy_from_slice(&samples[w_start..w_end]);
            } else {
                let copy_len = samples.len() - w_start;
                window_audio[..copy_len].copy_from_slice(&samples[w_start..]);
            }

            let batch_size = 1;
            let num_channels = 1;
            let input_tensor = Array3::from_shape_vec(
                (batch_size, num_channels, window_samples),
                window_audio,
            )?;

            let inputs = inputs![
                "x" => TensorRef::from_array_view(input_tensor.view())?
            ];

            let outputs = self.session.run(inputs)?;

            let logits_val = outputs
                .get("y")
                .ok_or_else(|| DiarizationError::OutputNotFound("y".to_string()))?;

            let logits_arr = logits_val.try_extract_array::<f32>()?;
            let logits_3d = logits_arr.to_owned().into_dimensionality::<ndarray::Ix3>()?;
            let shape = logits_3d.shape();
            let num_frames = shape[1];
            let num_classes = shape[2];

            if num_classes < 7 {
                return Err(DiarizationError::Shape(ndarray::ShapeError::from_kind(
                    ndarray::ErrorKind::IncompatibleShape,
                )));
            }

            if num_frames_per_window.is_none() {
                num_frames_per_window = Some(num_frames);
                let dt = config.segmentation_window_secs / num_frames as f64;
                delta_t = Some(dt);

                let estimated_size = (total_duration / dt).ceil() as usize + num_frames + 10;
                global_activity = vec![[0.0f32; 3]; estimated_size];
                global_weight = vec![0.0f32; estimated_size];
            }

            let dt = delta_t.unwrap();
            let w_start_time = w_start as f64 / sample_rate;
            let w_start_frame_float = w_start_time / dt;

            for f in 0..num_frames {
                // Softmax over 7 powerset classes
                let mut softmax = [0.0f32; 7];
                let mut max_val = f32::MIN;
                for c in 0..7 {
                    let val = logits_3d[[0, f, c]];
                    if val > max_val {
                        max_val = val;
                    }
                }
                let mut sum = 0.0f32;
                for c in 0..7 {
                    let exp_val = (logits_3d[[0, f, c]] - max_val).exp();
                    softmax[c] = exp_val;
                    sum += exp_val;
                }
                if sum > 0.0 {
                    for c in 0..7 {
                        softmax[c] /= sum;
                    }
                }

                // Marginalize powerset
                // Classes: 0: none, 1: A, 2: B, 3: C, 4: A+B, 5: A+C, 6: B+C
                let p_a = softmax[1] + softmax[4] + softmax[5];
                let p_b = softmax[2] + softmax[4] + softmax[6];
                let p_c = softmax[3] + softmax[5] + softmax[6];

                let global_idx = (w_start_frame_float + f as f64).round() as usize;

                if global_idx >= global_activity.len() {
                    global_activity.resize(global_idx + 1, [0.0f32; 3]);
                    global_weight.resize(global_idx + 1, 0.0f32);
                }

                global_activity[global_idx][0] += p_a;
                global_activity[global_idx][1] += p_b;
                global_activity[global_idx][2] += p_c;
                global_weight[global_idx] += 1.0;
            }
        }

        let dt = match delta_t {
            Some(val) => val,
            None => return Ok(Vec::new()),
        };

        for i in 0..global_activity.len() {
            let w = global_weight[i];
            if w > 0.0 {
                global_activity[i][0] /= w;
                global_activity[i][1] /= w;
                global_activity[i][2] /= w;
            }
        }

        let threshold = 0.5;
        let min_duration_secs = 0.3;
        let max_gap_secs = 0.1;

        let max_gap_frames = (max_gap_secs / dt).round() as usize;
        let min_duration_frames = (min_duration_secs / dt).round() as usize;

        let mut final_local_segments = Vec::new();

        for s in 0..3 {
            let mut is_active: Vec<bool> = global_activity
                .iter()
                .map(|probs| probs[s] >= threshold)
                .collect();

            let mut gap_start: Option<usize> = None;
            for i in 0..is_active.len() {
                if !is_active[i] {
                    if gap_start.is_none() {
                        gap_start = Some(i);
                    }
                } else {
                    if let Some(start) = gap_start {
                        let gap_len = i - start;
                        if gap_len <= max_gap_frames && start > 0 {
                            for j in start..i {
                                is_active[j] = true;
                            }
                        }
                        gap_start = None;
                    }
                }
            }

            let mut segment_start: Option<usize> = None;
            for i in 0..is_active.len() {
                if is_active[i] {
                    if segment_start.is_none() {
                        segment_start = Some(i);
                    }
                } else {
                    if let Some(start) = segment_start {
                        let len_frames = i - start;
                        if len_frames >= min_duration_frames {
                            let start_secs = start as f64 * dt;
                            let mut end_secs = i as f64 * dt;
                            if end_secs > total_duration {
                                end_secs = total_duration;
                            }
                            if start_secs < end_secs {
                                final_local_segments.push((start_secs, end_secs, s));
                            }
                        }
                        segment_start = None;
                    }
                }
            }
            if let Some(start) = segment_start {
                let len_frames = is_active.len() - start;
                if len_frames >= min_duration_frames {
                    let start_secs = start as f64 * dt;
                    let mut end_secs = is_active.len() as f64 * dt;
                    if end_secs > total_duration {
                        end_secs = total_duration;
                    }
                    if start_secs < end_secs {
                        final_local_segments.push((start_secs, end_secs, s));
                    }
                }
            }
        }

        final_local_segments.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        Ok(final_local_segments)
    }
}
