// Example usage:

mod data_loader;
mod factor_analysis;
use anyhow::Result;
use data_loader::*;
use factor_analysis::*;
use polars::prelude::*;
use std::sync::Arc;

fn main() -> Result<()> {
    let df = load_data()?;
    println!("{:?}", df);
    let parsed_df = parse_dataframe(df)?;
    println!("{:?}", parsed_df);
    let pivoted_df = pivot_dataframe(parsed_df)?;
    println!("{:?}", pivoted_df);
    let (returns, log_returns) = compute_returns(pivoted_df)?;
    println!("{:?}", returns);
    println!("{:?}", log_returns);
    Ok(())
}
