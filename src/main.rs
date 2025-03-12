mod analysis;
mod config;
mod data;
mod types;

use analysis::{
    factor_model::ThematicFactorModel, orthogonalization::FactorOrthogonalizer, pca::PCA,
    risk_attribution::RiskAttributor,
};
use data::loader::DataLoader;
use std::env;
use types::OrthogonalizationMethod;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load configuration
    let config = config::Config::load("config/factor_groups.yaml")?;
    println!(
        "Loaded factor configuration with {} groups",
        config.factor_groups.len()
    );

    // Get data file path from command line or use default
    let data_path = env::args()
        .nth(1)
        .unwrap_or_else(|| "data/prices.csv".to_string());

    println!("Loading data from: {}", data_path);
    let (_market_data, returns) = DataLoader::load_and_calculate_returns(&data_path)?;
    let tickers = DataLoader::get_tickers(&data_path)?;

    println!(
        "\nData shape: {} time periods × {} assets",
        returns.nrows(),
        returns.ncols()
    );

    // First, run PCA analysis with configured number of factors
    println!("\n=== Statistical Factor Analysis (PCA) ===");
    let pca = PCA::new(Some(config.model_settings.pca_factors));
    let mut pca_result = pca.fit_transform(returns.view())?;

    // Print explained variance ratios
    println!("\nExplained variance ratios:");
    let mut cumulative = 0.0;
    pca_result
        .explained_variance_ratio
        .iter()
        .enumerate()
        .for_each(|(i, ratio)| {
            cumulative += ratio;
            println!(
                "Factor {}: {:.4} (cumulative: {:.4})",
                i + 1,
                ratio,
                cumulative
            );
        });

    // Set asset list for PCA factors
    for group in pca_result.factor_model.get_factor_groups_mut() {
        group.assets = tickers.clone();
    }

    // Compute risk attribution for PCA factors
    println!("\n=== PCA Risk Attribution Analysis ===");
    let risk_attributor = RiskAttributor::new(
        pca_result.factor_model,
        config.model_settings.risk_lookback_days,
    );
    let attributions = risk_attributor.compute_portfolio_risk_attribution(
        returns.view(),
        &tickers,
        None, // No portfolio weights
    )?;

    // Print PCA risk attribution for each asset
    println!("\nPCA Risk Attribution by Asset:");
    for (i, attribution) in attributions.iter().enumerate() {
        if i < tickers.len() {
            println!("\n{}:", tickers[i]);
            println!("Total Risk: {:.2}%", attribution.total_risk * 100.0);
            println!("R-squared: {:.2}%", attribution.r_squared * 100.0);
            println!("Factor Contributions:");
            let mut contributions: Vec<_> = attribution.factor_contributions.iter().collect();
            contributions.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
            for (factor, contribution) in contributions {
                println!("  {}: {:.2}%", factor, contribution * 100.0);
            }
        }
    }

    // Now, let's do thematic factor analysis
    println!("\n=== Thematic Factor Analysis ===");
    let factor_groups = config.to_factor_groups();
    println!("\nDefined {} thematic factors:", factor_groups.len());
    for group in &factor_groups {
        println!("\n{}: {}", group.name, group.description);
        println!("Assets: {}", group.assets.join(", "));
    }

    let factor_model = ThematicFactorModel::new(factor_groups);
    let mut factor_returns = factor_model.compute_factor_returns(returns.view(), &tickers)?;

    // Orthogonalize factor returns if enabled
    if config.orthogonalization.enabled {
        println!(
            "\nOrthogonalizing factor returns using {} method...",
            match config.orthogonalization.method {
                OrthogonalizationMethod::GramSchmidt => "Gram-Schmidt",
                OrthogonalizationMethod::Pca => "PCA",
                OrthogonalizationMethod::Regression => "Regression",
            }
        );

        let mut orthogonalizer = FactorOrthogonalizer::new(
            config.orthogonalization.method.clone(),
            config.orthogonalization.constraints.max_correlation,
            config.orthogonalization.constraints.min_variance_explained,
        );

        let factor_names: Vec<String> = factor_model
            .get_factor_groups()
            .iter()
            .map(|g| g.name.clone())
            .collect();

        let priority_order = config.get_factor_priority();
        let (ortho_returns, kept_factors) =
            orthogonalizer.orthogonalize(factor_returns.view(), &factor_names, &priority_order);

        println!("\nKept {} orthogonalized factors:", kept_factors.len());
        for factor in &kept_factors {
            println!("  - {}", factor);
        }

        factor_returns = ortho_returns;
    }

    println!(
        "\nThematic Factor Returns Shape: {} periods × {} factors",
        factor_returns.nrows(),
        factor_returns.ncols()
    );

    // Compute risk attribution for thematic factors
    println!("\n=== Thematic Risk Attribution Analysis ===");
    let risk_attributor =
        RiskAttributor::new(factor_model, config.model_settings.risk_lookback_days);
    let attributions = risk_attributor.compute_portfolio_risk_attribution(
        returns.view(),
        &tickers,
        None, // No portfolio weights
    )?;

    // Print thematic risk attribution for each asset
    println!("\nThematic Risk Attribution by Asset:");
    for (i, attribution) in attributions.iter().enumerate() {
        if i < tickers.len() {
            println!("\n{}:", tickers[i]);
            println!("Total Risk: {:.2}%", attribution.total_risk * 100.0);
            println!("R-squared: {:.2}%", attribution.r_squared * 100.0);
            println!("Factor Contributions:");
            let mut contributions: Vec<_> = attribution.factor_contributions.iter().collect();
            contributions.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
            for (factor, contribution) in contributions {
                println!("  {}: {:.2}%", factor, contribution * 100.0);
            }
        }
    }

    Ok(())
}
