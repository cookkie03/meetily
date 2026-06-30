use crate::diarization::types::SpeakerEmbedding;

/// Clusters speaker embeddings using Agglomerative Hierarchical Clustering (AHC)
/// with Complete Linkage based on Cosine Distance.
///
/// Cosine distance is calculated as: D(u, v) = 1.0 - (u . v) for L2-normalized vectors.
pub fn cluster(
    embeddings: &[SpeakerEmbedding],
    threshold: f32,
) -> Vec<usize> {
    if embeddings.is_empty() {
        return Vec::new();
    }

    let n = embeddings.len();
    let mut labels = vec![0; n];
    
    // Start with each embedding in its own cluster.
    // Each cluster is represented by a vector of indices.
    let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

    loop {
        let num_clusters = clusters.len();
        if num_clusters <= 1 {
            break;
        }

        let mut min_dist = f32::MAX;
        let mut merge_i = 0;
        let mut merge_j = 0;

        // Find the pair of clusters with the smallest average-linkage distance (UPGMA)
        for i in 0..num_clusters {
            for j in (i + 1)..num_clusters {
                // Average linkage: mean distance between all pairs of elements in the two clusters
                let mut sum_dist = 0.0f32;
                let count = clusters[i].len() * clusters[j].len();
                
                for &idx_a in &clusters[i] {
                    for &idx_b in &clusters[j] {
                        let a = &embeddings[idx_a].data;
                        let b = &embeddings[idx_b].data;

                        // Cosine Similarity = dot product (since vectors are L2-normalized)
                        let similarity = a.dot(b);
                        // Cosine Distance = 1 - similarity
                        let dist = 1.0 - similarity;
                        sum_dist += dist;
                    }
                }

                let avg_dist = sum_dist / count as f32;

                if avg_dist < min_dist {
                    min_dist = avg_dist;
                    merge_i = i;
                    merge_j = j;
                }
            }
        }

        // If the closest clusters are closer than the threshold, merge them
        if min_dist < threshold {
            let mut elements = clusters.remove(merge_j);
            clusters[merge_i].append(&mut elements);
        } else {
            // No more clusters can be merged under the threshold
            break;
        }
    }

    // Assign a speaker ID (cluster index) to each audio segment
    for (cluster_id, cluster) in clusters.iter().enumerate() {
        for &idx in cluster {
            labels[idx] = cluster_id;
        }
    }

    labels
}
