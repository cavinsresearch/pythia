use risk_toolkit::{analysis::pca::PCA, data::loader::DataLoader};

#[test]
fn test_multi_asset_pca() {
    // Load test data
    let (_market_data, returns) =
        DataLoader::load_and_calculate_returns("tests/data/sample_prices.csv")
            .expect("Failed to load test data");

    // Create PCA with 2 components
    let pca = PCA::new(Some(2));
    let pca_result = pca.fit_transform(returns.view()).expect("PCA failed");

    // Basic assertions
    assert_eq!(returns.nrows(), 2); // 2 days of returns (3 days of prices - 1)
    assert_eq!(returns.ncols(), 2); // 2 tickers

    // Verify explained variance sums to approximately 1
    let total_variance: f64 = pca_result.explained_variance_ratio.sum();
    assert!((total_variance - 1.0).abs() < 1e-10);

    // Verify eigenvalues are sorted in descending order
    let mut prev = f64::INFINITY;
    for &val in pca_result.eigenvalues.iter() {
        assert!(val <= prev);
        prev = val;
    }
}
