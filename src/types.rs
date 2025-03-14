use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub enum OrthogonalizationMethod {
    GramSchmidt,
    Pca,
    Regression,
}
