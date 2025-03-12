use super::{DataError, MarketData, Result};
use chrono::NaiveDate;
use csv::ReaderBuilder;
use ndarray::Array2;
use std::collections::{HashMap, HashSet};
use std::path::Path;

pub struct DataLoader;

impl DataLoader {
    fn verify_required_columns(headers: &[String]) -> Result<()> {
        let required_columns = ["date", "ticker", "close"].map(String::from);
        let headers_set: HashSet<_> = headers.iter().map(|s| s.to_lowercase()).collect();

        for column in required_columns {
            if !headers_set.contains(&column.to_lowercase()) {
                return Err(DataError::MissingColumn(column));
            }
        }
        Ok(())
    }

    fn analyze_date_coverage(
        data_by_ticker: &HashMap<String, Vec<MarketData>>,
    ) -> (Vec<NaiveDate>, HashMap<String, f64>) {
        // Get all unique dates across all tickers
        let mut all_dates = HashSet::new();
        for data in data_by_ticker.values() {
            data.iter().for_each(|record| {
                all_dates.insert(record.date);
            });
        }
        let mut all_dates: Vec<_> = all_dates.into_iter().collect();
        all_dates.sort();

        // Calculate coverage percentage for each ticker
        let mut coverage = HashMap::new();
        for (ticker, data) in data_by_ticker {
            let ticker_dates: HashSet<_> = data.iter().map(|record| record.date).collect();
            let coverage_pct = ticker_dates.len() as f64 / all_dates.len() as f64 * 100.0;
            coverage.insert(ticker.clone(), coverage_pct);
        }

        (all_dates, coverage)
    }

    pub fn load_and_calculate_returns<P: AsRef<Path>>(
        path: P,
    ) -> Result<(Vec<MarketData>, Array2<f64>)> {
        let mut rdr = ReaderBuilder::new()
            .has_headers(true)
            .flexible(true)
            .trim(csv::Trim::All)
            .from_path(&path)?;

        // Verify required columns
        let headers: Vec<String> = rdr.headers()?.iter().map(|s| s.to_string()).collect();
        Self::verify_required_columns(&headers)?;

        // First pass: collect all data and organize by ticker
        let mut data_by_ticker: HashMap<String, Vec<MarketData>> = HashMap::new();
        for result in rdr.deserialize() {
            let record: MarketData = result?;
            data_by_ticker
                .entry(record.ticker.clone())
                .or_default()
                .push(record);
        }

        // Sort each ticker's data by date
        for data in data_by_ticker.values_mut() {
            data.sort_by(|a, b| a.date.cmp(&b.date));
        }

        // Analyze date coverage
        let (all_dates, coverage) = Self::analyze_date_coverage(&data_by_ticker);

        // Print coverage statistics
        println!("\nDate coverage analysis:");
        println!("Total unique dates: {}", all_dates.len());
        println!("Coverage by ticker:");
        let mut coverage_stats: Vec<_> = coverage.iter().collect();
        coverage_stats.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        for (ticker, coverage_pct) in coverage_stats {
            println!("{}: {:.1}%", ticker, coverage_pct);
        }

        // Find dates with most coverage (where most contracts trade)
        let threshold = 0.8; // Require at least 80% of tickers to have data
        let mut valid_dates = HashSet::new();
        for date in &all_dates {
            let tickers_with_data = data_by_ticker
                .values()
                .filter(|data| {
                    data.binary_search_by_key(date, |record| record.date)
                        .is_ok()
                })
                .count();
            let coverage = tickers_with_data as f64 / data_by_ticker.len() as f64;
            if coverage >= threshold {
                valid_dates.insert(*date);
            }
        }
        let mut valid_dates: Vec<_> = valid_dates.into_iter().collect();
        valid_dates.sort();

        println!(
            "\nFound {} dates with at least {}% ticker coverage",
            valid_dates.len(),
            threshold * 100.0
        );

        // Calculate returns matrix using only valid dates
        let n_dates = valid_dates.len() - 1;
        let n_tickers = data_by_ticker.len();
        let mut returns = Array2::zeros((n_dates, n_tickers));
        let mut tickers: Vec<_> = data_by_ticker.keys().cloned().collect();
        tickers.sort();

        for (j, ticker) in tickers.iter().enumerate() {
            let ticker_data = &data_by_ticker[ticker];
            let mut last_price = None;

            for (i, date) in valid_dates.windows(2).enumerate() {
                let current_date = date[1];
                let prev_date = date[0];

                // Find prices for both dates
                let current_price =
                    match ticker_data.binary_search_by_key(&current_date, |record| record.date) {
                        Ok(idx) => Some(ticker_data[idx].close),
                        Err(_) => None,
                    };

                let prev_price =
                    match ticker_data.binary_search_by_key(&prev_date, |record| record.date) {
                        Ok(idx) => Some(ticker_data[idx].close),
                        Err(_) => None,
                    };

                // Calculate return if we have both prices
                if let (Some(current), Some(prev)) = (current_price, prev_price) {
                    returns[[i, j]] = (current - prev) / prev;
                    last_price = Some(current);
                } else if let Some(last) = last_price {
                    // Use last available price for missing data
                    if let Some(current) = current_price {
                        returns[[i, j]] = (current - last) / last;
                        last_price = Some(current);
                    }
                }
            }
        }

        // Return the first ticker's data for date reference
        let market_data = data_by_ticker
            .remove(&tickers[0])
            .ok_or(DataError::MissingData)?;

        Ok((market_data, returns))
    }

    pub fn get_tickers<P: AsRef<Path>>(path: P) -> Result<Vec<String>> {
        let mut rdr = ReaderBuilder::new()
            .has_headers(true)
            .flexible(true)
            .trim(csv::Trim::All)
            .from_path(&path)?;

        let headers = rdr.headers()?;
        Self::verify_required_columns(&headers.iter().map(|s| s.to_string()).collect::<Vec<_>>())?;

        let mut tickers = HashSet::new();
        for result in rdr.deserialize() {
            let record: MarketData = result?;
            tickers.insert(record.ticker);
        }

        let mut tickers: Vec<_> = tickers.into_iter().collect();
        tickers.sort();
        Ok(tickers)
    }
}
