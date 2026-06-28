use std::path::{Path, PathBuf};
use tauri::{AppHandle, Emitter, Manager};
use crate::diarization::{Diarizer, DiarizationConfig};
use crate::state::AppState;

/// Ensures the diarization models are downloaded and returns their paths.
/// If they are missing, it downloads them from Hugging Face.
pub async fn ensure_diarization_models_internal<R: tauri::Runtime>(
    app: &AppHandle<R>,
) -> Result<(PathBuf, PathBuf), String> {
    let app_data_dir = app.path().app_data_dir()
        .map_err(|e| format!("Failed to get app data dir: {}", e))?;
    
    let diarization_dir = app_data_dir.join("models").join("diarization");
    if !diarization_dir.exists() {
        std::fs::create_dir_all(&diarization_dir)
            .map_err(|e| format!("Failed to create diarization models directory: {}", e))?;
    }

    let segmentation_path = diarization_dir.join("segmentation.onnx");
    let embedding_path = diarization_dir.join("embedding.onnx");

    // Check if segmentation model exists and has valid size (>1MB)
    let mut needs_segmentation = true;
    if segmentation_path.exists() {
        if let Ok(metadata) = std::fs::metadata(&segmentation_path) {
            if metadata.len() > 1_000_000 {
                needs_segmentation = false;
            }
        }
    }

    // Check if embedding model exists and has valid size (>10MB)
    let mut needs_embedding = true;
    if embedding_path.exists() {
        if let Ok(metadata) = std::fs::metadata(&embedding_path) {
            if metadata.len() > 10_000_000 {
                needs_embedding = false;
            }
        }
    }

    if !needs_segmentation && !needs_embedding {
        log::info!("Diarization models are already present.");
        return Ok((segmentation_path, embedding_path));
    }

    // Optimized HTTP client for downloads
    let client = reqwest::Client::builder()
        .tcp_nodelay(true)
        .connect_timeout(std::time::Duration::from_secs(30))
        .timeout(std::time::Duration::from_secs(1800))
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

    if needs_segmentation {
        let segmentation_url = "https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/resolve/main/model.onnx";
        download_file_with_progress(
            &client,
            segmentation_url,
            &segmentation_path,
            "segmentation",
            app,
        ).await?;
    }

    if needs_embedding {
        let embedding_url = "https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/main/wespeaker_en_voxceleb_resnet34.onnx";
        download_file_with_progress(
            &client,
            embedding_url,
            &embedding_path,
            "embedding",
            app,
        ).await?;
    }

    Ok((segmentation_path, embedding_path))
}

async fn download_file_with_progress<R: tauri::Runtime>(
    client: &reqwest::Client,
    url: &str,
    dest_path: &Path,
    model_name: &str,
    app: &AppHandle<R>,
) -> Result<(), String> {
    use futures_util::StreamExt;
    use tokio::io::AsyncWriteExt;

    log::info!("Downloading {} model from {}", model_name, url);

    let response = client.get(url)
        .send()
        .await
        .map_err(|e| format!("Failed to send download request for {}: {}", model_name, e))?;

    if !response.status().is_success() {
        return Err(format!("Download request failed for {} with status: {}", model_name, response.status()));
    }

    let total_size = response.content_length().unwrap_or(0);
    
    // Create destination file
    let mut file = tokio::fs::File::create(dest_path)
        .await
        .map_err(|e| format!("Failed to create file for {}: {}", model_name, e))?;

    let mut stream = response.bytes_stream();
    let mut downloaded: u64 = 0;
    
    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.map_err(|e| format!("Error reading stream chunk for {}: {}", model_name, e))?;
        file.write_all(&chunk).await
            .map_err(|e| format!("Failed to write chunk to file for {}: {}", model_name, e))?;
        
        downloaded += chunk.len() as u64;
        
        let percent = if total_size > 0 {
            ((downloaded as f64 / total_size as f64) * 100.0).min(100.0) as u8
        } else {
            0
        };

        // Emit progress event
        let _ = app.emit(
            "diarization-model-download-progress",
            serde_json::json!({
                "model": model_name,
                "progress": percent,
                "downloaded_bytes": downloaded,
                "total_bytes": total_size
            }),
        );
    }

    file.flush().await.map_err(|e| format!("Failed to flush file for {}: {}", model_name, e))?;
    log::info!("Successfully downloaded {} model", model_name);
    Ok(())
}

/// Runs speaker diarization on a decoded/resampled audio file and updates transcripts in the DB.
pub async fn run_diarization_internal<R: tauri::Runtime>(
    app: &AppHandle<R>,
    meeting_id: &str,
) -> Result<(), String> {
    log::info!("Starting diarization run for meeting: {}", meeting_id);

    let app_state = app.try_state::<AppState>()
        .ok_or_else(|| "App state not available".to_string())?;
    let pool = app_state.db_manager.pool();

    // 1. Get meeting folder path
    let folder_path_opt = sqlx::query_scalar::<_, Option<String>>(
        "SELECT folder_path FROM meetings WHERE id = ?"
    )
    .bind(meeting_id)
    .fetch_one(pool)
    .await
    .map_err(|e| format!("Database error fetching meeting: {}", e))?;

    let folder_path_str = folder_path_opt
        .ok_or_else(|| format!("Meeting {} has no associated folder path", meeting_id))?;

    let mut folder_path = PathBuf::from(&folder_path_str);
    if folder_path_str.starts_with("~/") {
        if let Some(home) = dirs::home_dir() {
            folder_path = home.join(&folder_path_str[2..]);
        }
    }

    let audio_path = folder_path.join("audio.mp4");
    if !audio_path.exists() {
        return Err(format!("Audio file not found at: {}", audio_path.display()));
    }

    let _ = app.emit(
        "diarization-progress",
        serde_json::json!({
            "meeting_id": meeting_id,
            "status": "downloading",
            "progress": 5,
            "message": "Ensuring diarization models are downloaded..."
        }),
    );

    // 2. Ensure models are present
    let (seg_model_path, emb_model_path) = ensure_diarization_models_internal(app).await?;

    let _ = app.emit(
        "diarization-progress",
        serde_json::json!({
            "meeting_id": meeting_id,
            "status": "decoding",
            "progress": 20,
            "message": "Decoding meeting audio..."
        }),
    );

    // 3. Decode audio to 16kHz mono f32 samples
    let audio_path_clone = audio_path.clone();
    let decoded = tokio::task::spawn_blocking(move || {
        crate::audio::decoder::decode_audio_file(&audio_path_clone)
    })
    .await
    .map_err(|e| format!("Decoding task panicked: {}", e))?
    .map_err(|e| format!("Audio decoding failed: {}", e))?;

    let audio_samples = tokio::task::spawn_blocking(move || {
        decoded.to_whisper_format()
    })
    .await
    .map_err(|e| format!("Resampling task panicked: {}", e))?;

    let _ = app.emit(
        "diarization-progress",
        serde_json::json!({
            "meeting_id": meeting_id,
            "status": "diarizing",
            "progress": 50,
            "message": "Running speaker diarization pipeline..."
        }),
    );

    // 4. Run speaker diarization pipeline
    let seg_model_clone = seg_model_path.clone();
    let emb_model_clone = emb_model_path.clone();
    let diar_segments = tokio::task::spawn_blocking(move || {
        let mut diarizer = Diarizer::new(&seg_model_clone, &emb_model_clone)?;
        let config = DiarizationConfig {
            segmentation_model_path: seg_model_clone,
            embedding_model_path: emb_model_clone,
            segmentation_window_secs: 10.0,
            segmentation_hop_secs: 0.5,
            clustering_threshold: 0.8,
            sample_rate: 16000,
        };
        diarizer.diarize(&audio_samples, &config)
    })
    .await
    .map_err(|e| format!("Diarization pipeline task panicked: {}", e))?
    .map_err(|e| format!("Diarization failed: {}", e))?;

    let _ = app.emit(
        "diarization-progress",
        serde_json::json!({
            "meeting_id": meeting_id,
            "status": "aligning",
            "progress": 85,
            "message": "Aligning speaker segments..."
        }),
    );

    // 5. Load transcription segments
    let transcripts = sqlx::query_as::<_, crate::database::models::Transcript>(
        "SELECT * FROM transcripts WHERE meeting_id = ? ORDER BY audio_start_time ASC"
    )
    .bind(meeting_id)
    .fetch_all(pool)
    .await
    .map_err(|e| format!("Failed to fetch transcripts: {}", e))?;

    // 6. Align and write to DB
    let mut tx = pool.begin().await
        .map_err(|e| format!("Failed to start database transaction: {}", e))?;

    for segment in transcripts {
        let mut best_speaker = None;
        let mut max_overlap = 0.0;
        let t_start = segment.audio_start_time.unwrap_or(0.0);
        let t_end = segment.audio_end_time.unwrap_or(0.0);

        for diar_seg in &diar_segments {
            let overlap_start = t_start.max(diar_seg.start);
            let overlap_end = t_end.min(diar_seg.end);
            let overlap = overlap_end - overlap_start;
            if overlap > max_overlap {
                max_overlap = overlap;
                best_speaker = Some(format!("Speaker {}", diar_seg.speaker_id + 1));
            }
        }

        sqlx::query("UPDATE transcripts SET speaker = ? WHERE id = ?")
            .bind(best_speaker)
            .bind(&segment.id)
            .execute(&mut *tx)
            .await
            .map_err(|e| format!("Failed to update speaker field: {}", e))?;
    }

    tx.commit().await
        .map_err(|e| format!("Failed to commit database updates: {}", e))?;

    log::info!("Diarization run complete for meeting: {}", meeting_id);
    Ok(())
}

/// Tauri command to trigger speaker diarization asynchronously.
#[tauri::command]
pub async fn run_diarization<R: tauri::Runtime>(
    app: AppHandle<R>,
    meeting_id: String,
) -> Result<(), String> {
    log::info!("run_diarization command triggered for meeting: {}", meeting_id);
    
    let app_clone = app.clone();
    let meeting_id_clone = meeting_id.clone();
    
    tokio::spawn(async move {
        let _ = app_clone.emit(
            "diarization-progress",
            serde_json::json!({
                "meeting_id": meeting_id_clone,
                "status": "started",
                "progress": 0,
                "message": "Starting diarization process..."
            }),
        );

        match run_diarization_internal(&app_clone, &meeting_id_clone).await {
            Ok(_) => {
                let _ = app_clone.emit(
                    "diarization-complete",
                    serde_json::json!({
                        "meeting_id": meeting_id_clone,
                        "status": "success",
                        "progress": 100
                    }),
                );
            }
            Err(e) => {
                log::error!("Diarization failed for {}: {}", meeting_id_clone, e);
                let _ = app_clone.emit(
                    "diarization-progress",
                    serde_json::json!({
                        "meeting_id": meeting_id_clone,
                        "status": "failed",
                        "progress": 0,
                        "message": format!("Diarization error: {}", e)
                    }),
                );
            }
        }
    });

    Ok(())
}
