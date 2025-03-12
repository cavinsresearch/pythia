use ndarray::{s, Array1, Array2, ArrayView2, Axis};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FactorType {
    // Traditional factors
    Thematic,
    Statistical,

    // Style factors
    Momentum {
        lookback_days: usize,
        skip_days: usize,
    },
    MeanReversion {
        lookback_days: usize,
    },
    Carry,

    // Macro factors
    RealRates,
    InflationSensitivity,
    GrowthSensitivity,

    // Market microstructure
    Liquidity {
        volume_lookback: usize,
    },
    MarketImpact {
        price_volume_ratio: bool,
    },
}

pub struct FactorBuilder;

impl FactorBuilder {
    pub fn compute_factor_returns(
        factor_type: &FactorType,
        returns: ArrayView2<f64>,
        metadata: Option<&HashMap<String, Array1<f64>>>,
    ) -> Array1<f64> {
        match factor_type {
            FactorType::Momentum {
                lookback_days,
                skip_days,
            } => Self::compute_momentum(returns, *lookback_days, *skip_days),
            FactorType::MeanReversion { lookback_days } => {
                Self::compute_mean_reversion(returns, *lookback_days)
            }
            FactorType::Carry => {
                if let Some(meta) = metadata {
                    if let Some(carry_data) = meta.get("carry") {
                        carry_data.clone()
                    } else {
                        Array1::zeros(returns.nrows())
                    }
                } else {
                    Array1::zeros(returns.nrows())
                }
            }
            FactorType::Liquidity { volume_lookback } => {
                if let Some(meta) = metadata {
                    if let Some(volume) = meta.get("volume") {
                        Self::compute_liquidity_factor(returns, volume, *volume_lookback)
                    } else {
                        Array1::zeros(returns.nrows())
                    }
                } else {
                    Array1::zeros(returns.nrows())
                }
            }
            _ => Array1::zeros(returns.nrows()),
        }
    }

    fn compute_momentum(
        returns: ArrayView2<f64>,
        lookback_days: usize,
        skip_days: usize,
    ) -> Array1<f64> {
        let n_periods = returns.nrows();
        let mut momentum = Array1::zeros(n_periods);

        // Skip the first lookback_days as we don't have enough data
        for i in (lookback_days + skip_days)..n_periods {
            let start_idx = i - lookback_days - skip_days;
            let end_idx = i - skip_days;

            // Calculate cumulative return over the lookback period
            let period_returns = returns.slice(s![start_idx..end_idx, ..]);
            let cum_returns = period_returns.sum_axis(Axis(0));

            // Momentum is the cross-sectional ranking of cumulative returns
            momentum[i] = cum_returns.mean().unwrap_or(0.0);
        }

        momentum
    }

    fn compute_mean_reversion(returns: ArrayView2<f64>, lookback_days: usize) -> Array1<f64> {
        let n_periods = returns.nrows();
        let mut mean_rev = Array1::zeros(n_periods);

        for i in lookback_days..n_periods {
            let start_idx = i - lookback_days;
            let period_returns = returns.slice(s![start_idx..i, ..]);

            // Calculate average return over lookback period
            let avg_returns = period_returns.mean_axis(Axis(0)).unwrap();

            // Mean reversion signal is negative of recent performance
            mean_rev[i] = -avg_returns.mean().unwrap_or(0.0);
        }

        mean_rev
    }

    fn compute_liquidity_factor(
        returns: ArrayView2<f64>,
        volume: &Array1<f64>,
        lookback: usize,
    ) -> Array1<f64> {
        let n_periods = returns.nrows();
        let mut liquidity = Array1::zeros(n_periods);

        for i in lookback..n_periods {
            let start_idx = i - lookback;
            let period_returns = returns.slice(s![start_idx..i, ..]);
            let period_volume = volume.slice(s![start_idx..i]);

            // Simple Amihud illiquidity ratio: |return| / volume
            let avg_ratio = period_returns.mapv(|x| x.abs()).mean_axis(Axis(0)).unwrap()
                / period_volume.mean().unwrap_or(1.0);

            // Take the mean of the ratios across assets
            liquidity[i] = -avg_ratio.mean().unwrap_or(0.0);
        }

        liquidity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_momentum_factor() {
        let returns = Array2::from_shape_vec(
            (10, 2),
            vec![
                0.01, 0.02, 0.01, 0.02, 0.01, 0.02, 0.01, 0.02, 0.01, 0.02, -0.01, -0.02, -0.01,
                -0.02, -0.01, -0.02, -0.01, -0.02, -0.01, -0.02,
            ],
        )
        .unwrap();

        let momentum = FactorBuilder::compute_momentum(
            returns.view(),
            5, // lookback
            0, // skip
        );

        assert_eq!(momentum.len(), 10);
    }
}
