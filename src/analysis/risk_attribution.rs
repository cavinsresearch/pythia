use super::factor_model::{FactorGroup, FactorModel, ThematicFactorModel};
use crate::types::OrthogonalizationMethod;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use ndarray_linalg::{Eigh, Solve, UPLO};
use ndarray_stats::QuantileExt;
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
pub struct RiskAttributor<M: FactorModel> {
    model: M,
    lookback_periods: usize,
    returns: Array2<f64>, // Store returns matrix
}

impl<M: FactorModel> RiskAttributor<M> {
    pub fn new(model: M, lookback_periods: usize, returns: Array2<f64>) -> Self {
        Self {
            model,
            lookback_periods,
            returns,
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

    /// Evaluate model fitness using expanding window analysis
    pub fn evaluate_model_fitness(
        &mut self,
        returns: ArrayView2<f64>,
        tickers: &[String],
        window_size: usize,
    ) -> Result<ModelFitMetrics> {
        let min_periods = window_size * 2;
        if returns.nrows() < min_periods {
            return Err(RiskAttributionError::InsufficientData);
        }

        let lambda = 0.05;
        let mut prediction_errors = Vec::new();
        let mut factor_returns_history = Vec::new();
        let step_size = 20;

        // Get factor names
        let factor_names: Vec<String> = self
            .model
            .get_factor_groups()
            .iter()
            .map(|g| g.name.clone())
            .collect();

        for i in (window_size..returns.nrows()).step_by(step_size) {
            let train_returns = returns.slice(s![i - window_size..i, ..]).to_owned();
            let test_returns = if i + step_size <= returns.nrows() {
                returns.slice(s![i..i + step_size, ..]).to_owned()
            } else {
                returns.slice(s![i..returns.nrows(), ..]).to_owned()
            };

            // Compute factor returns and store them
            let factor_returns = self
                .model
                .compute_factor_returns(train_returns.view(), tickers)?;
            factor_returns_history.push(factor_returns.clone());

            // Compute factor exposures for each asset
            let mut predictions = Array2::zeros((test_returns.nrows(), test_returns.ncols()));
            for j in 0..test_returns.ncols() {
                let asset_returns = train_returns.column(j);
                let exposures =
                    self.compute_factor_exposures(factor_returns.view(), asset_returns)?;
                let asset_predictions = factor_returns.dot(&exposures);
                predictions
                    .column_mut(j)
                    .assign(&asset_predictions.slice(s![..test_returns.nrows()]));
            }

            let error = &test_returns - &predictions;
            prediction_errors.extend(error.iter().cloned());
        }

        // Compute metrics
        let in_sample_r2 = self.compute_in_sample_r_squared(lambda)?;
        let out_sample_r2 = 1.0
            - prediction_errors.iter().map(|e| e * e).sum::<f64>()
                / returns.iter().skip(window_size).map(|r| r * r).sum::<f64>();

        let rmse = (prediction_errors.iter().map(|e| e * e).sum::<f64>()
            / prediction_errors.len() as f64)
            .sqrt();

        let mean = prediction_errors.iter().sum::<f64>() / prediction_errors.len() as f64;
        let variance = prediction_errors
            .iter()
            .map(|&e| (e - mean).powi(2))
            .sum::<f64>()
            / (prediction_errors.len() - 1) as f64;
        let ir = mean.abs() / variance.sqrt();

        // Compute factor stability for each factor
        let mut factor_stability = HashMap::new();

        for (j, factor_name) in factor_names.iter().enumerate() {
            // Extract factor returns time series for this factor
            let factor_series: Vec<f64> = factor_returns_history
                .iter()
                .map(|returns| returns.column(j).mean().unwrap_or(0.0))
                .collect();

            // Compute stability score
            let stability = self.compute_stability(&factor_series);
            factor_stability.insert(factor_name.clone(), stability);
        }

        Ok(ModelFitMetrics {
            in_sample_r_squared: in_sample_r2,
            out_of_sample_r_squared: out_sample_r2,
            factor_stability,
            prediction_error: rmse,
            information_ratio: ir,
        })
    }

    fn compute_stability(&self, series: &[f64]) -> f64 {
        if series.len() < 2 {
            return 0.0;
        }

        // Compute rolling correlations with exponential decay
        let decay: f64 = 0.94;
        let window_size = 20.min(series.len() / 2);
        let mut stability_scores = Vec::new();

        for i in window_size..series.len() {
            let window1 = &series[i - window_size..i];
            let window2 = if i + window_size <= series.len() {
                &series[i..i + window_size]
            } else {
                &series[i..series.len()]
            };

            // Skip if either window is too small
            if window2.len() < 2 {
                continue;
            }

            // Compute correlation between windows
            let mean1: f64 = window1.iter().sum::<f64>() / window1.len() as f64;
            let mean2: f64 = window2.iter().sum::<f64>() / window2.len() as f64;

            let var1: f64 = window1.iter().map(|x| (x - mean1).powi(2)).sum::<f64>();
            let var2: f64 = window2.iter().map(|x| (x - mean2).powi(2)).sum::<f64>();

            if var1 == 0.0 || var2 == 0.0 {
                continue;
            }

            let cov: f64 = window1
                .iter()
                .zip(window2.iter())
                .take(window2.len())
                .map(|(x, y)| (x - mean1) * (y - mean2))
                .sum::<f64>();

            let correlation = cov / (var1.sqrt() * var2.sqrt());

            // Apply exponential decay weight
            let weight = decay.powf((series.len() - i) as f64);
            stability_scores.push(correlation.abs() * weight);
        }

        if stability_scores.is_empty() {
            0.0
        } else {
            // Return weighted average of stability scores
            stability_scores.iter().sum::<f64>() / stability_scores.len() as f64
        }
    }

    fn compute_in_sample_r_squared(&self, lambda: f64) -> Result<f64> {
        let total_variance = self.returns.var(0.0);
        let residual_variance = self.compute_residual_variance(lambda)?;
        Ok(1.0 - (residual_variance / total_variance))
    }

    fn compute_residual_variance(&self, lambda: f64) -> Result<f64> {
        let n_periods = self.returns.nrows();
        let n_assets = self.returns.ncols();
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
            .compute_factor_returns(self.returns.view(), &tickers)?;

        // For each asset
        for i in 0..n_assets {
            let asset_returns = self.returns.column(i);
            let factor_exposures =
                self.compute_factor_exposures(factor_returns.view(), asset_returns)?;
            let fitted_returns = factor_returns.dot(&factor_exposures);
            let residuals = asset_returns.to_owned() - fitted_returns;
            total_residual_variance += residuals.var(0.0);
        }

        Ok(total_residual_variance / n_assets as f64)
    }

    /// Compute PCA factor loadings with regularization
    fn compute_factor_loadings(
        &self,
        returns: &Array2<f64>,
        lambda: f64,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        let cov = compute_covariance_matrix(returns.view(), self.lookback_periods)?;

        // Add regularization
        let n = cov.nrows();
        let mut reg_cov = cov.clone();
        for i in 0..n {
            reg_cov[[i, i]] += lambda;
        }

        // Compute eigendecomposition
        let (eigvals, eigvecs) = reg_cov
            .eigh(UPLO::Upper)
            .map_err(|e| RiskAttributionError::FactorModelError(e.into()))?;

        // Sort eigenvalues and eigenvectors in descending order
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&i, &j| eigvals[j].partial_cmp(&eigvals[i]).unwrap());

        let sorted_eigvals = Array1::from_vec(idx.iter().map(|&i| eigvals[i]).collect());
        let sorted_eigvecs = Array2::from_shape_vec(
            (n, n),
            idx.iter()
                .flat_map(|&i| eigvecs.column(i).to_vec())
                .collect(),
        )
        .map_err(|_| RiskAttributionError::InvalidDimensions)?;

        Ok((sorted_eigvecs, sorted_eigvals))
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
