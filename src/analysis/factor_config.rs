use super::factor_model::FactorGroup;
use super::factors::FactorType;
use super::weighting::WeightingScheme;

/// Creates predefined factor groups for market analysis
pub fn create_default_factor_groups() -> Vec<FactorGroup> {
    vec![
        FactorGroup {
            name: "Global Equity".to_string(),
            description: "Major global equity index futures".to_string(),
            assets: vec![
                "ES00-USA".to_string(),     // S&P 500
                "NQ00-USA".to_string(),     // NASDAQ 100
                "FESX00-EUR".to_string(),   // Euro STOXX 50
                "Z00-IFEU".to_string(),     // FTSE 100
                "NIK22500-OSE".to_string(), // Nikkei 225
            ],
            weights: None,
            weighting_scheme: Some(WeightingScheme::Equal),
            factor_type: FactorType::Thematic,
        },
        FactorGroup {
            name: "Energy".to_string(),
            description: "Global energy futures".to_string(),
            assets: vec![
                "CL00-USA".to_string(),   // US Light Crude
                "BRN00-IFEU".to_string(), // Brent Crude
                "NG00-USA".to_string(),   // Natural Gas
            ],
            weights: None,
            weighting_scheme: None,
            factor_type: FactorType::Thematic,
        },
        FactorGroup {
            name: "Precious Metals".to_string(),
            description: "Precious metals futures".to_string(),
            assets: vec![
                "GC00-USA".to_string(), // Gold
                "SI00-USA".to_string(), // Silver
            ],
            weights: None,
            weighting_scheme: None,
            factor_type: FactorType::Thematic,
        },
        FactorGroup {
            name: "US Rates".to_string(),
            description: "US Treasury futures across the curve".to_string(),
            assets: vec![
                "TU00-USA".to_string(), // 2Y
                "FV00-USA".to_string(), // 5Y
                "TY00-USA".to_string(), // 10Y
            ],
            weights: None,
            weighting_scheme: None,
            factor_type: FactorType::Thematic,
        },
        FactorGroup {
            name: "European Rates".to_string(),
            description: "European government bond futures".to_string(),
            assets: vec![
                "FGBS00-EUR".to_string(), // German 2Y
                "FGBM00-EUR".to_string(), // German 5Y
                "FGBL00-EUR".to_string(), // German 10Y
                "RLI00-IFEU".to_string(), // UK 10Y
            ],
            weights: None,
            weighting_scheme: None,
            factor_type: FactorType::Thematic,
        },
        FactorGroup {
            name: "Asian Rates".to_string(),
            description: "Asian government bond futures".to_string(),
            assets: vec![
                "JBT00-OSE".to_string(),  // Japanese 10Y
                "JGBS00-OSE".to_string(), // Japanese 20Y
            ],
            weights: None,
            weighting_scheme: None,
            factor_type: FactorType::Thematic,
        },
        FactorGroup {
            name: "Major FX".to_string(),
            description: "Major currency futures vs USD".to_string(),
            assets: vec![
                "EC00-USA".to_string(),  // EUR
                "JY00-USA".to_string(),  // JPY
                "BP00-USA".to_string(),  // GBP
                "SFC00-USA".to_string(), // CHF
            ],
            weights: None,
            weighting_scheme: None,
            factor_type: FactorType::Thematic,
        },
        FactorGroup {
            name: "EM FX".to_string(),
            description: "Emerging market currency futures".to_string(),
            assets: vec![
                "RMB00-USA".to_string(), // CNY
            ],
            weights: None,
            weighting_scheme: None,
            factor_type: FactorType::Thematic,
        },
    ]
}

/// Creates custom factor groups based on PCA results
pub fn create_pca_factor_groups(n_factors: usize, tickers: &[String]) -> Vec<FactorGroup> {
    (0..n_factors)
        .map(|i| FactorGroup {
            name: format!("PCA Factor {}", i + 1),
            description: format!("Statistical factor {} from PCA", i + 1),
            assets: tickers.to_vec(),
            weights: None,
            weighting_scheme: None,
            factor_type: FactorType::Statistical,
        })
        .collect()
}
