use super::factor_model::{FactorGroup, ThematicFactorModel};
use super::factors::FactorType;
use super::weighting::WeightingScheme;
use ndarray::{s, Array1, Array2, ArrayView2};
use ndarray_linalg::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PCAError {
    #[error("Linear algebra error: {0}")]
    LinAlgError(#[from] ndarray_linalg::error::LinalgError),
    #[error("Insufficient data for PCA")]
    InsufficientData,
}

pub type Result<T> = std::result::Result<T, PCAError>;

pub struct PCAResult {
    pub explained_variance_ratio: Array1<f64>,
    pub factor_model: ThematicFactorModel,
}

pub struct PCA {
    n_components: Option<usize>,
}

impl PCA {
    pub fn new(n_components: Option<usize>) -> Self {
        Self { n_components }
    }

    pub fn fit_transform(&self, data: ArrayView2<f64>) -> Result<PCAResult> {
        if data.nrows() < 2 || data.ncols() < 2 {
            return Err(PCAError::InsufficientData);
        }

        // Center the data
        let mean = data
            .mean_axis(ndarray::Axis(0))
            .ok_or(PCAError::InsufficientData)?;
        let centered = data.to_owned() - &mean.insert_axis(ndarray::Axis(0));

        // Compute covariance matrix
        let cov = centered.t().dot(&centered) / (data.nrows() - 1) as f64;

        // Compute eigendecomposition
        let (eigenvalues, eigenvectors) = cov.eigh(UPLO::Upper)?;

        // Sort eigenvalues and eigenvectors in descending order
        let mut indices: Vec<usize> = (0..eigenvalues.len()).collect();
        indices.sort_by(|&i, &j| eigenvalues[j].partial_cmp(&eigenvalues[i]).unwrap());

        let eigenvalues = Array1::from_vec(indices.iter().map(|&i| eigenvalues[i]).collect());

        let eigenvectors = Array2::from_shape_vec(
            (eigenvectors.nrows(), eigenvectors.ncols()),
            indices
                .iter()
                .flat_map(|&i| eigenvectors.column(i).to_vec())
                .collect(),
        )
        .unwrap();

        // Calculate explained variance ratio
        let total_variance = eigenvalues.sum();
        let explained_variance_ratio = &eigenvalues / total_variance;

        // Select number of components
        let n_components = self.n_components.unwrap_or_else(|| {
            explained_variance_ratio
                .iter()
                .take_while(|&&ratio| ratio > 0.1)
                .count()
        });

        // Create factor groups with normalized eigenvector weights
        let mut factor_groups = Vec::with_capacity(n_components);
        for i in 0..n_components {
            let mut weights = eigenvectors.column(i).to_vec();

            // Normalize weights to sum to 1
            let weight_sum = weights.iter().sum::<f64>();
            if weight_sum != 0.0 {
                for w in weights.iter_mut() {
                    *w /= weight_sum;
                }
            }

            factor_groups.push(FactorGroup {
                name: format!("PCA Factor {}", i + 1),
                description: format!(
                    "Statistical factor {} explaining {:.1}% of variance",
                    i + 1,
                    explained_variance_ratio[i] * 100.0
                ),
                assets: vec![], // Will be set by the caller
                weights: Some(weights),
                weighting_scheme: None,
                factor_type: FactorType::Statistical,
            });
        }

        Ok(PCAResult {
            explained_variance_ratio,
            factor_model: ThematicFactorModel::new(factor_groups),
        })
    }
}
