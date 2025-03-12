use super::factor_model::{FactorGroup, ThematicFactorModel};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::Serialize;
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RiskAttributionError {
    #[error("Factor model error: {0}")]
    FactorModelError(#[from] super::factor_model::FactorModelError),
    #[error("Invalid data dimensions")]
    InvalidDimensions,
}

pub type Result<T> = std::result::Result<T, RiskAttributionError>;

#[derive(Debug, Serialize)]
pub struct RiskAttribution {
    pub factor_contributions: HashMap<String, f64>, // Factor name -> risk contribution
    pub specific_risk: f64,                         // Idiosyncratic risk
    pub total_risk: f64,                            // Total risk
    pub r_squared: f64,                             // Model fit quality
}

#[derive(Debug)]
pub struct RiskAttributor {
    model: ThematicFactorModel,
    lookback_periods: usize, // Window for volatility calculation
}

impl RiskAttributor {
    pub fn new(model: ThematicFactorModel, lookback_periods: usize) -> Self {
        Self {
            model,
            lookback_periods,
        }
    }

    /// Compute risk attribution for a single asset
    pub fn compute_risk_attribution(
        &self,
        asset_returns: ArrayView1<f64>,
        factor_returns: ArrayView2<f64>,
        factor_groups: &[FactorGroup],
    ) -> Result<RiskAttribution> {
        // Compute factor exposures (betas)
        let betas = compute_regression_coefficients(factor_returns, asset_returns)?;

        // Compute factor covariance matrix
        let factor_cov = compute_covariance_matrix(factor_returns, self.lookback_periods)?;

        // Compute total variance explained by factors
        let systematic_var = betas.dot(&factor_cov.dot(&betas));

        // Compute residual variance
        let predicted_returns = factor_returns.dot(&betas);
        let residuals = asset_returns.to_owned() - predicted_returns;
        let specific_var = compute_variance(residuals.view(), self.lookback_periods)?;

        // Compute total variance and R-squared
        let total_var = systematic_var + specific_var;
        let total_risk = total_var.sqrt();
        let r_squared = systematic_var / total_var;

        // Compute individual factor contributions using betas (factor loadings)
        let mut factor_contributions = HashMap::new();

        // Store factor loadings (betas)
        for (i, group) in factor_groups.iter().enumerate() {
            factor_contributions.insert(group.name.clone(), betas[i]);
        }

        // Add idiosyncratic component
        factor_contributions.insert(
            "Idiosyncratic".to_string(),
            specific_var.sqrt() / total_risk,
        );

        Ok(RiskAttribution {
            factor_contributions,
            specific_risk: specific_var.sqrt(),
            total_risk,
            r_squared,
        })
    }

    /// Compute risk attribution for all assets
    pub fn compute_portfolio_risk_attribution(
        &self,
        returns: ArrayView2<f64>,
        tickers: &[String],
        weights: Option<&[f64]>,
    ) -> Result<Vec<RiskAttribution>> {
        // Compute factor returns and orthogonalize them
        let raw_factor_returns = self.model.compute_factor_returns(returns, tickers)?;
        let ortho_factor_returns = self
            .model
            .orthogonalize_factor_returns(raw_factor_returns.view())?;
        let factor_groups = self.model.get_factor_groups();

        let n_assets = returns.ncols();
        let mut attributions = Vec::with_capacity(n_assets);

        // Compute attribution for each asset using orthogonalized factors
        for i in 0..n_assets {
            let asset_returns = returns.column(i);
            let attribution = self.compute_risk_attribution(
                asset_returns,
                ortho_factor_returns.view(),
                factor_groups,
            )?;
            attributions.push(attribution);
        }

        // If weights are provided, compute weighted portfolio attribution
        if let Some(weights) = weights {
            if weights.len() != n_assets {
                return Err(RiskAttributionError::InvalidDimensions);
            }

            // Compute weighted average of attributions
            let mut portfolio_attribution = HashMap::new();
            let mut total_specific = 0.0;
            let mut total_risk = 0.0;

            for (i, attr) in attributions.iter().enumerate() {
                let weight = weights[i];
                for (factor, contrib) in &attr.factor_contributions {
                    *portfolio_attribution.entry(factor.clone()).or_insert(0.0) += weight * contrib;
                }
                total_specific += weight * attr.specific_risk;
                total_risk += weight * attr.total_risk;
            }

            attributions.push(RiskAttribution {
                factor_contributions: portfolio_attribution,
                specific_risk: total_specific,
                total_risk,
                r_squared: 1.0 - (total_specific / total_risk),
            });
        }

        Ok(attributions)
    }
}

/// Helper function to compute regression coefficients
fn compute_regression_coefficients(X: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<Array1<f64>> {
    use ndarray_linalg::solve::Solve;

    let xtx = X.t().dot(&X);
    let xty = X.t().dot(&y);

    xtx.solve(&xty)
        .map_err(|e| RiskAttributionError::FactorModelError(e.into()))
}

/// Helper function to compute covariance matrix
fn compute_covariance_matrix(returns: ArrayView2<f64>, lookback: usize) -> Result<Array2<f64>> {
    let n = returns.nrows();
    let k = returns.ncols();

    if n < lookback {
        return Err(RiskAttributionError::InvalidDimensions);
    }

    let mut cov = Array2::zeros((k, k));
    let recent_returns = returns.slice(ndarray::s![n - lookback.., ..]);

    for i in 0..k {
        for j in 0..=i {
            let cov_ij = compute_covariance(recent_returns.column(i), recent_returns.column(j))?;
            cov[[i, j]] = cov_ij;
            cov[[j, i]] = cov_ij; // Symmetric
        }
    }

    Ok(cov)
}

/// Helper function to compute covariance between two return series
fn compute_covariance(x: ArrayView1<f64>, y: ArrayView1<f64>) -> Result<f64> {
    let n = x.len();
    if n == 0 {
        return Err(RiskAttributionError::InvalidDimensions);
    }

    let mean_x = x.mean().unwrap();
    let mean_y = y.mean().unwrap();

    let cov = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
        .sum::<f64>()
        / (n as f64 - 1.0);

    Ok(cov)
}

/// Helper function to compute variance of a return series
fn compute_variance(x: ArrayView1<f64>, lookback: usize) -> Result<f64> {
    let n = x.len();
    if n < lookback {
        return Err(RiskAttributionError::InvalidDimensions);
    }

    let recent_x = x.slice(ndarray::s![n - lookback..]);
    compute_covariance(recent_x, recent_x)
}
