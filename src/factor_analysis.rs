use ndarray::{s, Array1, Array2};
use polars::prelude::*;
use std::collections::HashMap;

/// Converts wide-format price data to log returns
pub fn calculate_log_returns(df: &DataFrame) -> Result<DataFrame, PolarsError> {
    let mut return_df = df.clone();

    for col in df.get_columns() {
        if col.name() == "close" {
            let prices: Vec<f64> = df
                .column(col.name())?
                .f64()?
                .into_iter()
                .map(|x| x.unwrap_or(f64::NAN))
                .collect();

            let prices = Array1::from(prices);
            let log_prices = prices.mapv(f64::ln);
            let shifted = Array1::from(log_prices.slice(s![1..]).to_vec());
            let prev = Array1::from(log_prices.slice(s![..-1]).to_vec());
            let returns = (&shifted - &prev) * -1.0;

            let name: PlSmallStr = col.name().to_string().as_str().into();
            let mut return_series = Series::new(name, returns.to_vec());
            return_series = return_series.extend_constant(AnyValue::Null, 1)?;
            let return_col: Column = return_series.into_series().into();
            return_df.with_column(return_col)?;
        }
    }

    Ok(return_df.drop_nulls::<String>(None)?)
}

/// Standardizes returns (zero mean, unit variance) for each asset
pub fn standardize_returns(df: &DataFrame) -> Result<DataFrame, PolarsError> {
    let mut std_df = df.clone();

    for col in df.get_columns() {
        if col.name() != "date" {
            let values: Vec<f64> = df
                .column(col.name())?
                .f64()?
                .into_iter()
                .map(|x| x.unwrap_or(f64::NAN))
                .collect();

            let values = Array1::from(values);
            let mean = values.mean().unwrap_or(0.0);
            let std = values.std(0.0).max(1e-10);

            let standardized = (values - mean) / std;
            let name: PlSmallStr = col.name().to_string().as_str().into();
            let standardized_series = Series::new(name, standardized.to_vec());
            let std_col: Column = standardized_series.into_series().into();
            std_df.with_column(std_col)?;
        }
    }

    Ok(std_df)
}

/// Computes the correlation matrix of returns
pub fn compute_correlation_matrix(df: &DataFrame) -> Result<DataFrame, PolarsError> {
    let mut data: HashMap<String, Vec<f64>> = HashMap::new();
    let mut col_names: Vec<String> = Vec::new();

    // Extract numeric columns
    for col in df.get_columns() {
        if col.name() != "date" {
            let values: Vec<f64> = df
                .column(col.name())?
                .f64()?
                .into_iter()
                .map(|x| x.unwrap_or(f64::NAN))
                .collect();
            data.insert(col.name().to_string(), values);
            col_names.push(col.name().to_string());
        }
    }

    let n = col_names.len();
    let mut corr_matrix = Array2::zeros((n, n));

    // Compute correlation matrix
    for (i, name1) in col_names.iter().enumerate() {
        for (j, name2) in col_names.iter().enumerate() {
            let x = Array1::from(data.get(name1).unwrap().clone());
            let y = Array1::from(data.get(name2).unwrap().clone());

            let x_mean = x.mean().unwrap_or(0.0);
            let y_mean = y.mean().unwrap_or(0.0);
            let x_std = x.std(0.0).max(1e-10);
            let y_std = y.std(0.0).max(1e-10);

            let x_centered = &x - x_mean;
            let y_centered = &y - y_mean;
            let corr = (x_centered * &y_centered).sum() / (x_std * y_std * (x.len() as f64 - 1.0));

            corr_matrix[[i, j]] = corr;
        }
    }

    // Convert correlation matrix to DataFrame
    let mut corr_series = Vec::new();
    for (i, name) in col_names.iter().enumerate() {
        let row_values = corr_matrix.row(i).to_vec();
        let name: PlSmallStr = String::from(name).into();
        let series = Series::new(name, row_values);
        let col: Column = series.into_series().into();
        corr_series.push(col);
    }

    DataFrame::new(corr_series)
}

/// Creates rolling windows of returns for time-varying analysis
pub fn create_rolling_windows(
    df: &DataFrame,
    window_size: i64,
    step_size: i64,
) -> Result<Vec<DataFrame>, PolarsError> {
    let total_rows = df.height() as i64;
    let mut windows = Vec::new();

    for start in (0..total_rows - window_size).step_by(step_size as usize) {
        let window = df.slice(start, window_size as usize);
        windows.push(window);
    }

    Ok(windows)
}

/// Computes basic return statistics for each asset
pub fn compute_return_statistics(df: &DataFrame) -> Result<DataFrame, PolarsError> {
    let mut stats = Vec::new();

    for col in df.get_columns() {
        if col.name() != "date" {
            let values: Vec<f64> = df
                .column(col.name())?
                .f64()?
                .into_iter()
                .map(|x| x.unwrap_or(f64::NAN))
                .collect();

            let values = Array1::from(values);
            let mean = values.mean().unwrap_or(0.0);
            let std = values.std(0.0);

            // Calculate skewness
            let centered = &values - mean;
            let n = values.len() as f64;
            let skew = centered.mapv(|x| x.powi(3)).sum() / (n * std.powi(3));

            // Calculate kurtosis
            let kurt = centered.mapv(|x| x.powi(4)).sum() / (n * std.powi(4)) - 3.0;

            let name: PlSmallStr = col.name().to_string().as_str().into();
            let series = Series::new(name, &[mean, std, skew, kurt]);
            let col: Column = series.into_series().into();
            stats.push(col);
        }
    }

    DataFrame::new(stats)
}
