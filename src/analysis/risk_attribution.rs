use super::factor_model::{FactorGroup, ThematicFactorModel};
use crate::types::OrthogonalizationMethod;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use ndarray_linalg::solve::Solve;
use serde::Serialize;
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RiskAttributionError {
    #[error("Factor model error: {0}")]
    FactorModelError(#[from] super::factor_model::FactorModelError),
    #[error("Invalid data dimensions")]
    InvalidDimensions,
    #[error("Insufficient data for analysis")]
    InsufficientData,
}

pub type Result<T> = std::result::Result<T, RiskAttributionError>;

#[derive(Debug, Serialize)]
pub struct RiskAttribution {
    pub factor_contributions: HashMap<String, f64>, // Factor name -> risk contribution
    pub specific_risk: f64,                         // Idiosyncratic risk
    pub total_risk: f64,                            // Total risk
    pub r_squared: f64,                             // Model fit quality
}

#[derive(Debug, Serialize)]
pub struct ModelFitMetrics {
    pub in_sample_r_squared: f64,
    pub out_of_sample_r_squared: f64,
    pub factor_stability: HashMap<String, f64>,
    pub prediction_error: f64,
    pub information_ratio: f64,
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
        &mut self,
        returns: ArrayView2<f64>,
        tickers: &[String],
        weights: Option<&[f64]>,
    ) -> Result<Vec<RiskAttribution>> {
        // Compute factor returns and orthogonalize them
        let raw_factor_returns = self.model.compute_factor_returns(returns, tickers)?;
        let ortho_factor_returns = self.model.orthogonalize_factor_returns(
            raw_factor_returns.view(),
            OrthogonalizationMethod::GramSchmidt,
            0.3,  // max_correlation
            0.01, // min_variance_explained
        )?;
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

    /// Computes factor exposures (betas) for a given set of returns
    fn compute_factor_exposures(
        &self,
        factor_returns: ArrayView2<f64>,
        asset_returns: ArrayView1<f64>,
    ) -> Result<Array1<f64>> {
        let xtx = factor_returns.t().dot(&factor_returns);
        let xty = factor_returns.t().dot(&asset_returns);

        // Increase regularization term for more stability
        let n = xtx.nrows();
        let lambda = 0.01; // Increased from 1e-6
        let mut xtx_reg = xtx.to_owned();
        for i in 0..n {
            xtx_reg[[i, i]] += lambda;
        }

        xtx_reg
            .solve(&xty)
            .map_err(|e| RiskAttributionError::FactorModelError(e.into()))
    }

    /// Performs out-of-sample testing of the factor model using expanding window
    pub fn evaluate_model_fitness(
        &mut self,
        returns: ArrayView2<f64>,
        tickers: &[String],
        window_size: usize,
    ) -> Result<ModelFitMetrics> {
        let n_periods = returns.nrows();
        if n_periods < window_size * 2 {
            return Err(RiskAttributionError::InsufficientData);
        }

        let mut cumulative_prediction_error = 0.0;
        let mut cumulative_actual_return = 0.0;
        let mut factor_loadings_history: HashMap<String, Vec<f64>> = HashMap::new();
        let mut prediction_count = 0;

        // Initialize factor loadings history
        for group in self.model.get_factor_groups() {
            factor_loadings_history.insert(group.name.clone(), Vec::new());
        }

        // Expanding window analysis with step size
        let step_size = 20; // Predict every 20 days to reduce noise
        for end_idx in (window_size..n_periods).step_by(step_size) {
            // Training window
            let train_returns = returns.slice(s![end_idx - window_size..end_idx, ..]);

            // Compute factor returns for training period
            let factor_returns = self
                .model
                .compute_factor_returns(train_returns.view(), tickers)?;

            // Store factor loadings for stability analysis
            for (i, group) in self.model.get_factor_groups().iter().enumerate() {
                if let Some(loadings) = factor_loadings_history.get_mut(&group.name) {
                    loadings.push(factor_returns[[factor_returns.nrows() - 1, i]]);
                }
            }

            // Predict next period returns for each asset
            let next_period_returns = if end_idx + step_size <= n_periods {
                returns.slice(s![end_idx..end_idx + step_size, ..])
            } else {
                returns.slice(s![end_idx.., ..])
            };

            let mut total_squared_error = 0.0;
            let mut total_actual_return = 0.0;

            for asset_idx in 0..returns.ncols() {
                let asset_returns = train_returns.column(asset_idx);
                let factor_exposures =
                    self.compute_factor_exposures(factor_returns.view(), asset_returns)?;

                // Predict returns for each period in the step
                for period in 0..next_period_returns.nrows() {
                    let predicted_return = factor_returns
                        .row(factor_returns.nrows() - 1)
                        .dot(&factor_exposures);
                    let actual_return = next_period_returns[[period, asset_idx]];

                    total_squared_error += (predicted_return - actual_return).powi(2);
                    total_actual_return += actual_return;
                    prediction_count += 1;
                }
            }

            cumulative_prediction_error += total_squared_error;
            cumulative_actual_return += total_actual_return;
        }

        // Calculate factor stability (exponentially weighted autocorrelation)
        let mut factor_stability = HashMap::new();
        for (factor_name, loadings) in factor_loadings_history.iter() {
            let stability = self.compute_stability(loadings);
            factor_stability.insert(factor_name.clone(), stability);
        }

        // Compute final metrics
        let prediction_error = (cumulative_prediction_error / prediction_count as f64).sqrt();
        let information_ratio =
            cumulative_actual_return / (prediction_error * (prediction_count as f64).sqrt());

        Ok(ModelFitMetrics {
            in_sample_r_squared: self.compute_in_sample_r_squared(returns)?,
            out_of_sample_r_squared: 1.0 - (prediction_error.powi(2) / returns.var(0.0)),
            factor_stability,
            prediction_error,
            information_ratio,
        })
    }

    fn compute_stability(&self, series: &[f64]) -> f64 {
        if series.len() < 2 {
            return 0.0;
        }

        let decay: f64 = 0.94; // Exponential decay factor
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        let mut weight_sum = 0.0;

        for i in 1..series.len() {
            let weight = decay.powf((series.len() - i) as f64);
            numerator += weight * series[i] * series[i - 1];
            denominator += weight * series[i].powi(2);
            weight_sum += weight;
        }

        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    fn compute_in_sample_r_squared(&self, returns: ArrayView2<f64>) -> Result<f64> {
        let total_variance = returns.var(0.0);
        let residual_variance = self.compute_residual_variance(returns)?;
        Ok(1.0 - (residual_variance / total_variance))
    }

    fn compute_residual_variance(&self, returns: ArrayView2<f64>) -> Result<f64> {
        let n_periods = returns.nrows();
        let n_assets = returns.ncols();
        let mut total_residual_variance = 0.0;

        // Get tickers from the model's factor groups
        let tickers: Vec<String> = self
            .model
            .get_factor_groups()
            .iter()
            .flat_map(|g| g.assets.clone())
            .collect();

        // Compute factor returns for the entire period
        let factor_returns = self
            .model
            .compute_factor_returns(returns.view(), &tickers)?;

        // For each asset
        for i in 0..n_assets {
            let asset_returns = returns.column(i);
            let factor_exposures =
                self.compute_factor_exposures(factor_returns.view(), asset_returns)?;
            let fitted_returns = factor_returns.dot(&factor_exposures);
            let residuals = asset_returns.to_owned() - fitted_returns;
            total_residual_variance += residuals.var(0.0);
        }

        Ok(total_residual_variance / n_assets as f64)
    }
}

/// Helper function to compute regression coefficients
fn compute_regression_coefficients(X: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<Array1<f64>> {
    use ndarray_linalg::solve::Solve;

    let xtx = X.t().dot(&X);
    let xty = X.t().dot(&y);

    // Add a small regularization term (ridge regression)
    let n = xtx.nrows();
    let lambda = 1e-6; // Small regularization parameter
    let mut xtx_reg = xtx.to_owned();
    for i in 0..n {
        xtx_reg[[i, i]] += lambda;
    }

    xtx_reg
        .solve(&xty)
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
