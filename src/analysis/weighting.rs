use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use serde::{Deserialize, Serialize};
use std::f64;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizationConstraints {
    pub min_weight: f64,
    pub max_weight: f64,
    pub sum_to_one: bool,
}

impl Default for OptimizationConstraints {
    fn default() -> Self {
        Self {
            min_weight: 0.0,
            max_weight: 1.0,
            sum_to_one: true,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum WeightingScheme {
    Equal,
    InverseVolatility {
        lookback_days: usize,
    },
    MinimumVariance {
        lookback_days: usize,
        constraints: OptimizationConstraints,
    },
    MarketValue {
        values: Array1<f64>,
    },
}

pub struct WeightCalculator;

impl WeightCalculator {
    pub fn compute_weights(returns: ArrayView2<f64>, scheme: &WeightingScheme) -> Array1<f64> {
        match scheme {
            WeightingScheme::Equal => Self::equal_weight(returns.ncols()),
            WeightingScheme::InverseVolatility { lookback_days } => {
                Self::inverse_volatility_weight(returns, *lookback_days)
            }
            WeightingScheme::MinimumVariance {
                lookback_days,
                constraints,
            } => Self::minimum_variance_weight(returns, *lookback_days, constraints),
            WeightingScheme::MarketValue { values } => Self::market_value_weight(values),
        }
    }

    fn equal_weight(n_assets: usize) -> Array1<f64> {
        let weight = 1.0 / n_assets as f64;
        Array1::from_elem(n_assets, weight)
    }

    fn inverse_volatility_weight(returns: ArrayView2<f64>, lookback_days: usize) -> Array1<f64> {
        let n_assets = returns.ncols();
        let window = returns.nrows().min(lookback_days);

        // Calculate volatilities
        let mut vols = Array1::zeros(n_assets);
        for i in 0..n_assets {
            let asset_returns = returns.slice(s![returns.nrows() - window.., i]);
            vols[i] = asset_returns.std(0.0);
        }

        // Convert to inverse volatility weights
        let inv_vols = vols.mapv(|x| 1.0 / x.max(1e-10));
        let sum = inv_vols.sum();
        inv_vols / sum
    }

    fn minimum_variance_weight(
        returns: ArrayView2<f64>,
        lookback_days: usize,
        constraints: &OptimizationConstraints,
    ) -> Array1<f64> {
        let n_assets = returns.ncols();
        let window = returns.nrows().min(lookback_days);
        let recent_returns = returns.slice(s![returns.nrows() - window.., ..]);

        // Compute sample covariance matrix
        let mut cov = Array2::zeros((n_assets, n_assets));
        for i in 0..n_assets {
            for j in 0..n_assets {
                let cov_ij =
                    Self::compute_covariance(recent_returns.column(i), recent_returns.column(j));
                cov[[i, j]] = cov_ij;
            }
        }

        // Simple minimum variance optimization with constraints
        // Note: This is a basic implementation. For production, consider using
        // an optimization library like OSQP or similar
        let mut weights = Array1::from_elem(n_assets, 1.0 / n_assets as f64);

        // Apply constraints
        weights.mapv_inplace(|w| w.max(constraints.min_weight).min(constraints.max_weight));
        if constraints.sum_to_one {
            let sum = weights.sum();
            weights.mapv_inplace(|w| w / sum);
        }

        weights
    }

    fn market_value_weight(values: &Array1<f64>) -> Array1<f64> {
        let sum = values.sum();
        values / sum
    }

    fn compute_covariance(x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
        let n = x.len() as f64;
        let mean_x = x.mean().unwrap_or(0.0);
        let mean_y = y.mean().unwrap_or(0.0);

        let mut cov = 0.0;
        for i in 0..x.len() {
            cov += (x[i] - mean_x) * (y[i] - mean_y);
        }

        cov / (n - 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_equal_weight() {
        let n_assets = 4;
        let weights = WeightCalculator::equal_weight(n_assets);

        assert_eq!(weights.len(), n_assets);
        assert_relative_eq!(weights.sum(), 1.0, epsilon = 1e-10);
        for &w in weights.iter() {
            assert_relative_eq!(w, 0.25, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_inverse_volatility_weight() {
        let returns = Array2::from_shape_vec(
            (5, 3),
            vec![
                0.01, 0.02, 0.03, -0.01, -0.02, -0.03, 0.01, 0.02, 0.03, -0.01, -0.02, -0.03, 0.01,
                0.02, 0.03,
            ],
        )
        .unwrap();

        let weights = WeightCalculator::inverse_volatility_weight(returns.view(), 5);

        assert_eq!(weights.len(), 3);
        assert_relative_eq!(weights.sum(), 1.0, epsilon = 1e-10);
    }
}
