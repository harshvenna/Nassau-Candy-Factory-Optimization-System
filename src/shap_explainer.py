"""
Nassau Candy SupplyChainAI — SHAP Explainability Module
========================================================
Provides global and local SHAP explanations for both ML targets.
Outputs plotly-compatible DataFrames for the Streamlit dashboard.
"""

import numpy as np
import pandas as pd
import shap
import warnings
warnings.filterwarnings("ignore")


def _get_explainer(model, X_background: np.ndarray):
    """
    Auto-select the right SHAP explainer class.
    TreeExplainer for tree models, KernelExplainer as fallback.
    """
    model_name = type(model).__name__
    tree_types = (
        "RandomForestRegressor", "GradientBoostingRegressor",
        "XGBRegressor", "LGBMRegressor", "CatBoostRegressor",
        "DecisionTreeRegressor", "ExtraTreesRegressor",
    )
    if model_name in tree_types:
        return shap.TreeExplainer(model)
    else:
        # Linear or unknown: use LinearExplainer / KernelExplainer
        try:
            return shap.LinearExplainer(model, X_background)
        except Exception:
            bg = shap.sample(X_background, min(100, len(X_background)))
            return shap.KernelExplainer(model.predict, bg)


def compute_shap_values(
    model,
    X: np.ndarray,
    feature_names: list,
    max_samples: int = 500,
) -> dict:
    """
    Compute SHAP values on a (optionally sub-sampled) dataset.

    Returns dict with:
      - shap_values: ndarray (n_samples, n_features)
      - shap_df: DataFrame with mean |SHAP| per feature (global importance)
      - X_sample: the input samples used
    """
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    explainer   = _get_explainer(model, X_sample)
    shap_values = explainer.shap_values(X_sample)

    # For multi-output models (rare here), take first output
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    mean_abs = np.abs(shap_values).mean(axis=0)
    shap_df  = pd.DataFrame({
        "Feature":          feature_names,
        "Mean_Abs_SHAP":    mean_abs,
    }).sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
    shap_df["Importance_Pct"] = (
        shap_df["Mean_Abs_SHAP"] / shap_df["Mean_Abs_SHAP"].sum() * 100
    ).round(2)

    return {
        "shap_values": shap_values,
        "shap_df":     shap_df,
        "X_sample":    X_sample,
        "feature_names": feature_names,
    }


def local_explanation(
    model,
    X_instance: np.ndarray,
    X_background: np.ndarray,
    feature_names: list,
) -> pd.DataFrame:
    """
    SHAP explanation for a single prediction instance.
    Returns a DataFrame with feature contributions.
    """
    explainer   = _get_explainer(model, X_background)
    shap_vals   = explainer.shap_values(X_instance.reshape(1, -1))

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]

    shap_vals = shap_vals.flatten()
    df = pd.DataFrame({
        "Feature":      feature_names,
        "SHAP_Value":   shap_vals,
        "Feature_Value": X_instance.flatten(),
    }).sort_values("SHAP_Value", key=abs, ascending=False).reset_index(drop=True)

    df["Direction"] = df["SHAP_Value"].apply(lambda x: "Increases" if x > 0 else "Decreases")
    return df


def interpret_shap_globally(shap_df: pd.DataFrame, target_name: str) -> list:
    """
    Generate plain-English business interpretations for the top features.
    Returns list of insight strings.
    """
    insights = []
    for _, row in shap_df.head(5).iterrows():
        feat = row["Feature"].replace("_", " ")
        pct  = row["Importance_Pct"]
        insights.append(
            f"**{feat}** explains {pct:.1f}% of {target_name} predictions — "
            f"one of the most influential drivers in the model."
        )
    return insights
