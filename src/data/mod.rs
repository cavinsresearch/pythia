pub mod loader;

use chrono::NaiveDate;
use serde::Deserialize;
use thiserror::Error;

#[derive(Debug, Deserialize)]
pub struct MarketData {
    pub date: NaiveDate,
    pub ticker: String,
    pub close: f64,
    #[serde(default)]
    #[serde(skip_deserializing)]
    _other: (), // This will allow extra fields in the CSV
}

#[derive(Debug, Error)]
pub enum DataError {
    #[error("CSV error: {0}")]
    Csv(#[from] csv::Error),
    #[error("Date parse error: {0}")]
    DateParse(#[from] chrono::ParseError),
    #[error("Missing data for calculation")]
    MissingData,
    #[error("Missing required column: {0}")]
    MissingColumn(String),
}

pub type Result<T> = std::result::Result<T, DataError>;
