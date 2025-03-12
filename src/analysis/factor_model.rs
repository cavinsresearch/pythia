use crate::analysis::{
    factors::{FactorBuilder, FactorType},
    orthogonalization::FactorOrthogonalizer,
    weighting::{WeightCalculator, WeightingScheme},
};
use crate::types::OrthogonalizationMethod;
use ndarray::{Array1, Array2, ArrayView2, Axis};
use ndarray_linalg::error::LinalgError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum FactorModelError {
    #[error("Missing asset: {0}")]
    MissingAsset(String),
    #[error("Computation error: {0}")]
    ComputationError(String),
}

impl From<LinalgError> for FactorModelError {
    fn from(err: LinalgError) -> Self {
        FactorModelError::ComputationError(err.to_string())
    }
}

pub type Result<T> = std::result::Result<T, FactorModelError>;

/// Common trait for all factor models
pub trait FactorModel {
    /// Compute factor returns for the given returns data
    fn compute_factor_returns(
        &self,
        returns: ArrayView2<f64>,
        tickers: &[String],
    ) -> Result<Array2<f64>>;

    /// Orthogonalize factor returns using the specified method
    fn orthogonalize_factor_returns(
        &mut self,
        factor_returns: ArrayView2<f64>,
        method: OrthogonalizationMethod,
        max_correlation: f64,
        min_variance_explained: f64,
    ) -> Result<Array2<f64>>;

    /// Get the factor groups that define this model
    fn get_factor_groups(&self) -> &[FactorGroup];

    /// Get mutable access to factor groups
    fn get_factor_groups_mut(&mut self) -> &mut [FactorGroup];

    /// Add metadata that can be used in factor computations
    fn add_metadata(&mut self, key: &str, data: Array1<f64>);
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorGroup {
    pub name: String,
    pub description: String,
    pub assets: Vec<String>,
    pub weights: Option<Vec<f64>>,
    pub weighting_scheme: Option<WeightingScheme>,
    pub factor_type: FactorType,
}

#[derive(Debug)]
pub struct ThematicFactorModel {
    factor_groups: Vec<FactorGroup>,
    metadata: HashMap<String, Array1<f64>>,
}

impl ThematicFactorModel {
    pub fn new(factor_groups: Vec<FactorGroup>) -> Self {
        Self {
            factor_groups,
            metadata: HashMap::new(),
        }
    }
}

impl FactorModel for ThematicFactorModel {
    fn compute_factor_returns(
        &self,
        returns: ArrayView2<f64>,
        tickers: &[String],
    ) -> Result<Array2<f64>> {
        let n_periods = returns.nrows();
        let n_factors = self.factor_groups.len();
        let mut factor_returns = Array2::zeros((n_periods, n_factors));

        for (factor_idx, group) in self.factor_groups.iter().enumerate() {
            match group.factor_type {
                FactorType::Thematic => {
                    // Get indices of assets in this group
                    let mut group_indices = Vec::new();
                    for asset in &group.assets {
                        if let Some(pos) = tickers.iter().position(|x| x == asset) {
                            group_indices.push(pos);
                        } else {
                            return Err(FactorModelError::MissingAsset(asset.clone()));
                        }
                    }

                    // Extract returns for assets in this group
                    let group_returns = returns.select(Axis(1), &group_indices);

                    // Compute weights based on scheme or use provided weights
                    let weights = if let Some(scheme) = &group.weighting_scheme {
                        WeightCalculator::compute_weights(group_returns.view(), scheme)
                    } else if let Some(w) = &group.weights {
                        Array1::from(w.clone())
                    } else {
                        // Default to equal weight
                        WeightCalculator::compute_weights(
                            group_returns.view(),
                            &WeightingScheme::Equal,
                        )
                    };

                    // Compute weighted average returns for this factor
                    for period in 0..n_periods {
                        let period_returns: Array1<f64> = group_indices
                            .iter()
                            .map(|&idx| returns[[period, idx]])
                            .collect();

                        factor_returns[[period, factor_idx]] = period_returns.dot(&weights);
                    }
                }
                FactorType::Statistical => {
                    // For statistical factors, use the weights from the factor group
                    if let Some(weights) = &group.weights {
                        // Add weights to metadata for the factor builder
                        let mut metadata = self.metadata.clone();
                        metadata.insert("weights".to_string(), Array1::from(weights.clone()));

                        // Compute factor returns using the weights
                        let factor_returns_vec = FactorBuilder::compute_factor_returns(
                            &group.factor_type,
                            returns,
                            Some(&metadata),
                        );

                        for (i, &val) in factor_returns_vec.iter().enumerate() {
                            factor_returns[[i, factor_idx]] = val;
                        }
                    } else {
                        // If no weights provided, use equal weighting
                        let n_assets = returns.ncols();
                        let equal_weight = 1.0 / n_assets as f64;
                        for t in 0..n_periods {
                            factor_returns[[t, factor_idx]] = returns.row(t).sum() * equal_weight;
                        }
                    }
                }
                _ => {
                    // Use the FactorBuilder for other factor types
                    let factor_returns_vec = FactorBuilder::compute_factor_returns(
                        &group.factor_type,
                        returns,
                        Some(&self.metadata),
                    );

                    for (i, &val) in factor_returns_vec.iter().enumerate() {
                        factor_returns[[i, factor_idx]] = val;
                    }
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
    ) -> Result<Array2<f64>> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_thematic_factor_returns() {
        let factor_groups = vec![FactorGroup {
            name: "Test Group".to_string(),
            description: "Test description".to_string(),
            assets: vec!["Asset1".to_string(), "Asset2".to_string()],
            weights: None,
            weighting_scheme: Some(WeightingScheme::Equal),
            factor_type: FactorType::Thematic,
        }];

        let model = ThematicFactorModel::new(factor_groups);

        let returns =
            Array2::from_shape_vec((3, 2), vec![0.01, 0.02, -0.01, -0.02, 0.01, 0.02]).unwrap();

        let tickers = vec!["Asset1".to_string(), "Asset2".to_string()];

        let factor_returns = model
            .compute_factor_returns(returns.view(), &tickers)
            .unwrap();

        assert_eq!(factor_returns.shape(), &[3, 1]);
        assert_relative_eq!(factor_returns[[0, 0]], 0.015, epsilon = 1e-10);
    }
}
