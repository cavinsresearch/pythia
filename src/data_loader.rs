use anyhow::Result;
use polars::prelude::*;
use polars_ops::pivot::{pivot, PivotAgg};

pub fn load_data() -> Result<DataFrame> {
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some("data/prices.csv".into()))
        .unwrap()
        .finish()
        .unwrap();
    Ok(df)
}

pub fn parse_dataframe(df: DataFrame) -> Result<DataFrame> {
    let df = df.select(vec![
        "date", "ticker", "open", "high", "low", "close", "volume",
    ])?;
    Ok(df)
}

// Pivot the dataframe so that the ticker is the column header, date is the index, and close is the value.
pub fn pivot_dataframe(df: DataFrame) -> Result<DataFrame> {
    let df = pivot(
        &df,
        ["ticker"],
        Some(["date"]),
        Some(["close"]),
        true,
        Some(PivotAgg::First),
        None,
    )?;
    Ok(df)
}

// Compute 1-day return and 1-day log return from a pivoted dataframe.
// Return 2 dataframes with ticker as the column header, date as the index, and return and log_return as the values respectively.
pub fn compute_returns(df: DataFrame) -> Result<(DataFrame, DataFrame)> {
    // Clone the original dataframe for separate modifications
    let mut returns_df = df.clone();
    let mut log_returns_df = df.clone();

    // Iterate over all columns. Skip the 'date' column which serves as the index.
    for col in df.get_column_names() {
        if col == "date" {
            continue;
        }
        // Get the column as a f64 series
        let series = df.column(col)?.f64()?.clone();
        // Shift the series by 1
        let shifted = series.shift(1);

        // Compute returns: (current / previous) - 1
        let ret_series = (&series / &shifted) - 1.0;
        returns_df.replace(col, ret_series.into_series())?;

        // Compute log returns: ln(current / previous)
        // Use apply to handle potential null values
        let ratio_series = &series / &shifted;
        let log_ret_series = ratio_series.apply(|v: Option<f64>| v.map(|x| x.ln()));
        log_returns_df.replace(col, log_ret_series.into_series())?;
    }

    Ok((returns_df, log_returns_df))
}
