use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

use crate::analysis::factor_model::FactorGroup;

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelSettings {
    pub pca_factors: usize,
    pub risk_lookback_days: usize,
    pub min_coverage_pct: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FactorGroupConfig {
    pub name: String,
    pub description: String,
    pub assets: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    pub factor_groups: Vec<FactorGroupConfig>,
    pub model_settings: ModelSettings,
}

impl Config {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = fs::read_to_string(path)?;
        let config: Config = serde_yaml::from_str(&contents)?;
        Ok(config)
    }

    pub fn to_factor_groups(&self) -> Vec<FactorGroup> {
        self.factor_groups
            .iter()
            .map(|group| FactorGroup {
                name: group.name.clone(),
                description: group.description.clone(),
                assets: group.assets.clone(),
                weights: None,
            })
            .collect()
    }
}
