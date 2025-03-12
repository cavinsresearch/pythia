use super::orthogonalization::FactorOrthogonalizer;
use crate::types::OrthogonalizationMethod;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum FactorModelError {
    #[error("Linear algebra error: {0}")]
    LinAlgError(#[from] ndarray_linalg::error::LinalgError),
    #[error("Invalid factor group: {0}")]
    InvalidFactorGroup(String),
    #[error("Missing asset in factor group: {0}")]
    MissingAsset(String),
}

pub type Result<T> = std::result::Result<T, FactorModelError>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorGroup {
    pub name: String,
    pub description: String,
    pub assets: Vec<String>,
    pub weights: Option<Vec<f64>>, // Optional custom weights for assets
}

#[derive(Debug)]
pub struct ThematicFactorModel {
    factor_groups: Vec<FactorGroup>,
    asset_to_group: HashMap<String, usize>, // Maps assets to their primary group index
}

impl ThematicFactorModel {
    pub fn new(factor_groups: Vec<FactorGroup>) -> Self {
        let mut asset_to_group = HashMap::new();

        // Build asset to group mapping
        for (group_idx, group) in factor_groups.iter().enumerate() {
            for asset in &group.assets {
                asset_to_group.insert(asset.clone(), group_idx);
            }
        }

        Self {
            factor_groups,
            asset_to_group,
        }
    }

    /// Compute factor returns based on thematic groupings
    pub fn compute_factor_returns(
        &self,
        returns: ArrayView2<f64>,
        tickers: &[String],
    ) -> Result<Array2<f64>> {
        let n_periods = returns.nrows();
        let n_factors = self.factor_groups.len();
        let mut factor_returns = Array2::zeros((n_periods, n_factors));

        for (factor_idx, group) in self.factor_groups.iter().enumerate() {
            // Get indices of assets in this group
            let mut group_indices = Vec::new();
            for asset in &group.assets {
                if let Some(pos) = tickers.iter().position(|x| x == asset) {
                    group_indices.push(pos);
                } else {
                    return Err(FactorModelError::MissingAsset(asset.clone()));
                }
            }

            // Get weights for this group
            let weights = match &group.weights {
                Some(w) => Array1::from(w.clone()),
                None => {
                    // Equal weights if not specified
                    let weight = 1.0 / group_indices.len() as f64;
                    Array1::from_vec(vec![weight; group_indices.len()])
                }
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

        Ok(factor_returns)
    }

    /// Get factor exposures (betas) for each asset
    pub fn compute_factor_exposures(
        &self,
        returns: ArrayView2<f64>,
        tickers: &[String],
    ) -> Result<Array2<f64>> {
        let factor_returns = self.compute_factor_returns(returns, tickers)?;

        // Prepare matrices for regression
        let X = factor_returns.view();
        let n_assets = returns.ncols();
        let n_factors = self.factor_groups.len();
        let mut betas = Array2::zeros((n_assets, n_factors));

        // For each asset, regress returns on factor returns
        for asset_idx in 0..n_assets {
            let y = returns.slice(ndarray::s![.., asset_idx]);
            let beta = compute_regression_coefficients(X.view(), y.view())?;
            betas.row_mut(asset_idx).assign(&beta);
        }

        Ok(betas)
    }

    /// Get factor groups
    pub fn get_factor_groups(&self) -> &[FactorGroup] {
        &self.factor_groups
    }

    /// Get mutable factor groups
    pub fn get_factor_groups_mut(&mut self) -> &mut Vec<FactorGroup> {
        &mut self.factor_groups
    }

    pub fn orthogonalize_factor_returns(
        &self,
        factor_returns: ArrayView2<f64>,
    ) -> Result<Array2<f64>> {
        let mut orthogonalizer = FactorOrthogonalizer::new(
            OrthogonalizationMethod::GramSchmidt,
            0.8,  // max_correlation threshold
            0.01, // min_variance_explained threshold
        );

        // Create factor names and priority order
        let factor_names: Vec<String> = self.factor_groups.iter().map(|g| g.name.clone()).collect();

        // Use same order for priority
        let priority_order = factor_names.clone();

        // Call orthogonalize and return just the factor returns
        let (ortho_returns, _) =
            orthogonalizer.orthogonalize(factor_returns, &factor_names, &priority_order);
        Ok(ortho_returns)
    }
}

/// Helper function to compute regression coefficients
fn compute_regression_coefficients(X: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<Array1<f64>> {
    use ndarray_linalg::solve::Solve;

    // Add constant term (intercept)
    let n_obs = X.nrows();
    let mut X_with_constant = Array2::ones((n_obs, X.ncols() + 1));
    X_with_constant.slice_mut(ndarray::s![.., 1..]).assign(&X);

    // Compute (X'X)^(-1)X'y
    let xtx = X_with_constant.t().dot(&X_with_constant);
    let xty = X_with_constant.t().dot(&y);

    let coefficients = xtx.solve(&xty).map_err(FactorModelError::LinAlgError)?;

    // Return only the factor coefficients (exclude intercept)
    Ok(coefficients.slice(ndarray::s![1..]).to_owned())
}
