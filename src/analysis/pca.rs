use super::factor_model::{
    FactorGroup, FactorModel, FactorModelError, Result as ModelResult, ThematicFactorModel,
};
use super::factors::FactorType;
use super::weighting::WeightingScheme;
use crate::types::OrthogonalizationMethod;
use ndarray::{s, Array1, Array2, ArrayView2};
use ndarray_linalg::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PCAError {
    #[error("Linear algebra error: {0}")]
    LinAlgError(#[from] ndarray_linalg::error::LinalgError),
    #[error("Insufficient data for PCA")]
    InsufficientData,
}

pub type Result<T> = std::result::Result<T, PCAError>;

#[derive(Debug)]
pub struct StatisticalFactorModel {
    factor_groups: Vec<FactorGroup>,
    metadata: HashMap<String, Array1<f64>>,
    explained_variance_ratio: Array1<f64>,
}

impl StatisticalFactorModel {
    pub fn new(factor_groups: Vec<FactorGroup>, explained_variance_ratio: Array1<f64>) -> Self {
        Self {
            factor_groups,
            metadata: HashMap::new(),
            explained_variance_ratio,
        }
    }

    pub fn get_explained_variance_ratio(&self) -> &Array1<f64> {
        &self.explained_variance_ratio
    }
}

impl FactorModel for StatisticalFactorModel {
    fn compute_factor_returns(
        &self,
        returns: ArrayView2<f64>,
        tickers: &[String],
    ) -> ModelResult<Array2<f64>> {
        let n_periods = returns.nrows();
        let n_factors = self.factor_groups.len();
        let mut factor_returns = Array2::zeros((n_periods, n_factors));

        for (factor_idx, group) in self.factor_groups.iter().enumerate() {
            if let Some(weights) = &group.weights {
                let weights = Array1::from(weights.clone());
                // Compute weighted returns for each time period
                for t in 0..n_periods {
                    let period_returns = returns.row(t);
                    factor_returns[[t, factor_idx]] = period_returns.dot(&weights);
                }
            }
        }

        Ok(factor_returns)
    }

    fn orthogonalize_factor_returns(
        &mut self,
        factor_returns: ArrayView2<f64>,
        method: OrthogonalizationMethod,
        max_correlation: f64,
        min_variance_explained: f64,
    ) -> ModelResult<Array2<f64>> {
        use super::orthogonalization::FactorOrthogonalizer;

        let mut orthogonalizer =
            FactorOrthogonalizer::new(method, max_correlation, min_variance_explained);

        let factor_names: Vec<String> = self.factor_groups.iter().map(|g| g.name.clone()).collect();
        let priority_order: Vec<String> = factor_names.clone();

        let (ortho_returns, _kept_factors) =
            orthogonalizer.orthogonalize(factor_returns, &factor_names, &priority_order);

        Ok(ortho_returns)
    }

    fn get_factor_groups(&self) -> &[FactorGroup] {
        &self.factor_groups
    }

    fn get_factor_groups_mut(&mut self) -> &mut [FactorGroup] {
        &mut self.factor_groups
    }

    fn add_metadata(&mut self, key: &str, data: Array1<f64>) {
        self.metadata.insert(key.to_string(), data);
    }
}

pub struct PCA {
    n_components: Option<usize>,
}

impl PCA {
    pub fn new(n_components: Option<usize>) -> Self {
        Self { n_components }
    }

    pub fn fit_transform(&self, data: ArrayView2<f64>) -> Result<StatisticalFactorModel> {
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

        Ok(StatisticalFactorModel::new(
            factor_groups,
            explained_variance_ratio,
        ))
    }
}
