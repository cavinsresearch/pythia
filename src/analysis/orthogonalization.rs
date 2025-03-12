use crate::types::OrthogonalizationMethod;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum OrthogonalizationError {
    #[error("Invalid data dimensions")]
    InvalidDimensions,
    #[error("Linear algebra error: {0}")]
    LinAlgError(#[from] ndarray_linalg::error::LinalgError),
}

pub type Result<T> = std::result::Result<T, OrthogonalizationError>;

#[derive(Debug)]
pub struct FactorOrthogonalizer {
    pub method: OrthogonalizationMethod,
    pub max_correlation: f64,
    pub min_variance_explained: f64,
}

impl FactorOrthogonalizer {
    pub fn new(
        method: OrthogonalizationMethod,
        max_correlation: f64,
        min_variance_explained: f64,
    ) -> Self {
        Self {
            method,
            max_correlation,
            min_variance_explained,
        }
    }

    pub fn orthogonalize(
        &mut self,
        factor_returns: ArrayView2<f64>,
        factor_names: &[String],
        priority_order: &[String],
    ) -> (Array2<f64>, Vec<String>) {
        match self.method {
            OrthogonalizationMethod::GramSchmidt => {
                self.gram_schmidt_orthogonalization(factor_returns, factor_names, priority_order)
            }
            OrthogonalizationMethod::Pca => {
                self.pca_orthogonalization(factor_returns, factor_names, priority_order)
            }
            OrthogonalizationMethod::Regression => {
                // For now, fall back to Gram-Schmidt
                self.gram_schmidt_orthogonalization(factor_returns, factor_names, priority_order)
            }
        }
    }

    fn gram_schmidt_orthogonalization(
        &self,
        factor_returns: ArrayView2<f64>,
        factor_names: &[String],
        priority_order: &[String],
    ) -> (Array2<f64>, Vec<String>) {
        let n_obs = factor_returns.nrows();
        let n_factors = factor_returns.ncols();

        // Create mapping from factor name to column index
        let factor_indices: HashMap<_, _> = factor_names
            .iter()
            .enumerate()
            .map(|(i, name)| (name.as_str(), i))
            .collect();

        // Initialize orthogonalized returns with zeros
        let mut ortho_returns = Array2::zeros((n_obs, n_factors));
        let mut kept_factors: Vec<String> = Vec::new();

        // Process factors in priority order
        for factor_name in priority_order {
            if let Some(&idx) = factor_indices.get(factor_name.as_str()) {
                let mut current_returns = factor_returns.column(idx).to_owned();
                let original_returns = current_returns.clone();

                // Orthogonalize against all previously kept factors
                for prev_factor in &kept_factors {
                    if let Some(&prev_idx) = factor_indices.get(prev_factor.as_str()) {
                        let prev_returns = ortho_returns.column(prev_idx);
                        let proj = self.project(&current_returns.view(), &prev_returns);
                        current_returns = &current_returns - &proj;
                    }
                }

                // Check if factor meets minimum variance threshold
                let var_explained = self.compute_relative_variance_explained(
                    &current_returns.view(),
                    &original_returns.view(),
                );
                if var_explained >= self.min_variance_explained {
                    ortho_returns
                        .slice_mut(s![.., idx])
                        .assign(&current_returns);
                    kept_factors.push(factor_name.clone());
                }
            }
        }

        // Add any remaining factors not in priority order
        for (name, &idx) in &factor_indices {
            let factor_name = name.to_string();
            if !kept_factors.contains(&factor_name) {
                let mut current_returns = factor_returns.column(idx).to_owned();
                let original_returns = current_returns.clone();

                // Orthogonalize against all kept factors
                for prev_factor in &kept_factors {
                    if let Some(&prev_idx) = factor_indices.get(prev_factor.as_str()) {
                        let prev_returns = ortho_returns.column(prev_idx);
                        let proj = self.project(&current_returns.view(), &prev_returns);
                        current_returns = &current_returns - &proj;
                    }
                }

                let var_explained = self.compute_relative_variance_explained(
                    &current_returns.view(),
                    &original_returns.view(),
                );
                if var_explained >= self.min_variance_explained {
                    ortho_returns
                        .slice_mut(s![.., idx])
                        .assign(&current_returns);
                    kept_factors.push(factor_name);
                }
            }
        }

        (ortho_returns, kept_factors)
    }

    fn project(&self, v: &ArrayView1<f64>, u: &ArrayView1<f64>) -> Array1<f64> {
        let dot = v.dot(u);
        let norm_sq = u.dot(u);
        if norm_sq > 0.0 {
            let scalar = dot / norm_sq;
            u * scalar
        } else {
            Array1::zeros(v.len())
        }
    }

    fn compute_variance_explained(&self, returns: &ArrayView1<f64>) -> f64 {
        let n = returns.len() as f64;
        let mean = returns.sum() / n;
        let var = returns
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<f64>()
            / n;
        var
    }

    fn compute_relative_variance_explained(
        &self,
        returns: &ArrayView1<f64>,
        original_returns: &ArrayView1<f64>,
    ) -> f64 {
        let ortho_var = self.compute_variance_explained(returns);
        let orig_var = self.compute_variance_explained(original_returns);
        if orig_var > 0.0 {
            ortho_var / orig_var
        } else {
            0.0
        }
    }

    fn pca_orthogonalization(
        &self,
        factor_returns: ArrayView2<f64>,
        factor_names: &[String],
        priority_order: &[String],
    ) -> (Array2<f64>, Vec<String>) {
        // TODO: Implement PCA-based orthogonalization
        // For now, fall back to Gram-Schmidt
        self.gram_schmidt_orthogonalization(factor_returns, factor_names, priority_order)
    }
}
