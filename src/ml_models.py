"""
Nassau Candy SupplyChainAI — Machine Learning Module
=====================================================
Trains, evaluates, and persists predictive models for:
  1. Lead Time (regression)
  2. Gross Profit (regression)

Models: Linear Regression, Random Forest, Gradient Boosting,
        XGBoost, LightGBM, CatBoost

Evaluation: MAE, RMSE, R², 5-fold CV
Hyperparameter tuning via RandomizedSearchCV.
Best model is exported to models/ for use by the Streamlit app.
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from typing import Tuple

from sklearn.linear_model    import LinearRegression, Ridge
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline        import Pipeline

import xgboost  as xgb
import lightgbm as lgb
from catboost  import CatBoostRegressor

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# FEATURE DEFINITIONS
# ─────────────────────────────────────────

# Features used to predict Lead Time
LEAD_TIME_FEATURES = [
    "ShipMode_Days",
    "Shipping_Distance_km",
    "Factory_Load_Pct",
    "Demand_Score",
    "Region_Demand",
    "Order_Month",
    "Order_Quarter",
    "Order_DayOfWeek",
    "Product_Profit_Rank",
    "Product_Sales_Rank",
    "Factory_Efficiency",
    "Route_Risk",
    "LeadTime_Variability",
    # Encoded categoricals (added during preprocessing below)
    "Factory_enc",
    "Region_enc",
    "Ship_Mode_enc",
    "Division_enc",
]

# Features used to predict Gross Profit
PROFIT_FEATURES = [
    "Sales",
    "Units",
    "Cost",
    "Shipping_Distance_km",
    "Lead_Time",
    "ShipMode_Days",
    "Factory_Load_Pct",
    "Demand_Score",
    "Region_Demand",
    "Order_Month",
    "Order_Quarter",
    "Customer_Value",
    "Cost_Ratio",
    "Sales_Per_Unit",
    "Cost_Per_Unit",
    "Profit_Stability",
    "Factory_Profitability",
    "Factory_Efficiency",
    "Product_Profit_Rank",
    "Product_Sales_Rank",
    "Factory_enc",
    "Region_enc",
    "Ship_Mode_enc",
    "Division_enc",
]

TARGET_LEADTIME = "Lead_Time"
TARGET_PROFIT   = "Gross Profit"

MODELS_DIR = "models"


# ─────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Label-encode Factory, Region, Ship Mode, Division.
    Returns (encoded_df, encoders_dict).
    """
    df = df.copy()
    encoders = {}

    for col, enc_col in [
        ("Factory",   "Factory_enc"),
        ("Region",    "Region_enc"),
        ("Ship Mode", "Ship_Mode_enc"),
        ("Division",  "Division_enc"),
    ]:
        le = LabelEncoder()
        df[enc_col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders


def prepare_X_y(df: pd.DataFrame, feature_cols: list, target: str):
    """Return (X, y) after dropping rows with NaN in required cols."""
    needed = feature_cols + [target]
    sub = df[needed].dropna()
    X = sub[feature_cols].values.astype(float)
    y = sub[target].values.astype(float)
    return X, y


# ─────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────

def _get_model_suite(task: str = "lead_time"):
    """
    Returns dict of {name: model_instance} for the given task.
    task: 'lead_time' | 'profit'
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression":  Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_leaf=3,
            n_jobs=-1, random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1, random_state=42, verbosity=0
        ),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1, random_state=42, verbose=-1
        ),
        "CatBoost": CatBoostRegressor(
            iterations=200, depth=6, learning_rate=0.05,
            random_seed=42, verbose=False
        ),
    }
    return models


# ─────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────

def evaluate_model(model, X_train, X_test, y_train, y_test, cv_folds=5) -> dict:
    """Fit model, return comprehensive metrics dict."""
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    # Cross-validation on training set
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=cv_folds, scoring="neg_mean_absolute_error", n_jobs=-1
    )

    return {
        "MAE_train":   mean_absolute_error(y_train, y_pred_train),
        "MAE_test":    mean_absolute_error(y_test,  y_pred_test),
        "RMSE_train":  np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "RMSE_test":   np.sqrt(mean_squared_error(y_test,  y_pred_test)),
        "R2_train":    r2_score(y_train, y_pred_train),
        "R2_test":     r2_score(y_test,  y_pred_test),
        "CV_MAE_mean": -cv_scores.mean(),
        "CV_MAE_std":  cv_scores.std(),
    }


def train_and_compare(X, y, task: str = "lead_time") -> Tuple[dict, object, str]:
    """
    Trains all models, compares performance, returns:
      (results_dict, best_model, best_model_name)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models  = _get_model_suite(task)
    results = {}

    print(f"\n{'='*60}")
    print(f"  Training {task.replace('_',' ').title()} Models")
    print(f"{'='*60}")

    for name, model in models.items():
        print(f"  ▸ {name:<25}", end="", flush=True)
        try:
            metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
            results[name] = {**metrics, "model": model, "status": "ok"}
            print(f"  MAE={metrics['MAE_test']:.3f}  R²={metrics['R2_test']:.3f}")
        except Exception as exc:
            results[name] = {"status": "error", "error": str(exc)}
            print(f"  ERROR: {exc}")

    # Select best by lowest test MAE
    valid   = {k: v for k, v in results.items() if v["status"] == "ok"}
    best_nm = min(valid, key=lambda k: valid[k]["MAE_test"])
    best_md = valid[best_nm]["model"]

    print(f"\n  ★ Best model: {best_nm}  (MAE={valid[best_nm]['MAE_test']:.3f})")

    return results, best_md, best_nm


# ─────────────────────────────────────────
# HYPERPARAMETER TUNING
# ─────────────────────────────────────────

def tune_best_model(best_model, X, y, n_iter: int = 30) -> object:
    """
    RandomizedSearchCV on the best model.
    Only XGBoost / LightGBM / RF have defined search spaces here.
    Others are returned as-is.
    """
    model_name = type(best_model).__name__
    param_distributions = None

    if "XGB" in model_name:
        param_distributions = {
            "n_estimators":    [100, 200, 300, 400],
            "max_depth":       [3, 5, 6, 8],
            "learning_rate":   [0.01, 0.03, 0.05, 0.1],
            "subsample":       [0.6, 0.8, 1.0],
            "colsample_bytree":[0.6, 0.8, 1.0],
        }
    elif "LGBM" in model_name:
        param_distributions = {
            "n_estimators":  [100, 200, 300],
            "max_depth":     [4, 6, 8, -1],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample":     [0.6, 0.8, 1.0],
        }
    elif "RandomForest" in model_name:
        param_distributions = {
            "n_estimators": [100, 200, 300, 400],
            "max_depth":    [6, 8, 10, None],
            "min_samples_leaf": [1, 2, 3, 5],
        }

    if param_distributions is None:
        print("  Tuning: no search space defined for this model type — skipping.")
        return best_model

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
    rscv = RandomizedSearchCV(
        best_model, param_distributions,
        n_iter=n_iter, cv=3, scoring="neg_mean_absolute_error",
        n_jobs=-1, random_state=42, verbose=0
    )
    rscv.fit(X_train, y_train)
    tuned = rscv.best_estimator_
    y_pred = tuned.predict(X_val)
    print(f"  Tuned MAE: {mean_absolute_error(y_val, y_pred):.3f}")
    print(f"  Best params: {rscv.best_params_}")
    return tuned


# ─────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────

def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """
    Returns a DataFrame of feature importances sorted descending.
    Works for tree-based models; returns None for linear models.
    """
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_)
    else:
        return None

    df = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": imp,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    df["Importance_Pct"] = (df["Importance"] / df["Importance"].sum() * 100).round(2)
    return df


# ─────────────────────────────────────────
# PERSISTENCE
# ─────────────────────────────────────────

def save_model(obj, filename: str, directory: str = MODELS_DIR) -> str:
    """Pickle an object to models/ directory."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  Saved → {path}")
    return path


def load_model(filename: str, directory: str = MODELS_DIR):
    """Load a pickled object."""
    path = os.path.join(directory, filename)
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────
# FULL TRAINING PIPELINE
# ─────────────────────────────────────────

def run_full_training(df: pd.DataFrame) -> dict:
    """
    End-to-end training for both targets.
    Returns artefact dict with models, encoders, metrics, feature importance.
    """
    df_enc, encoders = encode_categoricals(df)

    artefacts = {"encoders": encoders}

    # ── Target 1: Lead Time ─────────────────
    print("\n" + "─" * 60)
    print(" TARGET 1: LEAD TIME PREDICTION")
    print("─" * 60)

    X_lt, y_lt = prepare_X_y(df_enc, LEAD_TIME_FEATURES, TARGET_LEADTIME)
    results_lt, best_lt, name_lt = train_and_compare(X_lt, y_lt, task="lead_time")

    print(f"\n  Tuning {name_lt}...")
    best_lt_tuned = tune_best_model(best_lt, X_lt, y_lt, n_iter=20)

    artefacts["lead_time"] = {
        "model":            best_lt_tuned,
        "model_name":       name_lt,
        "features":         LEAD_TIME_FEATURES,
        "results":          {k: {m: v for m, v in r.items() if m != "model"}
                             for k, r in results_lt.items()},
        "feature_importance": get_feature_importance(best_lt_tuned, LEAD_TIME_FEATURES),
    }

    # ── Target 2: Gross Profit ──────────────
    print("\n" + "─" * 60)
    print(" TARGET 2: GROSS PROFIT PREDICTION")
    print("─" * 60)

    X_pr, y_pr = prepare_X_y(df_enc, PROFIT_FEATURES, TARGET_PROFIT)
    results_pr, best_pr, name_pr = train_and_compare(X_pr, y_pr, task="profit")

    print(f"\n  Tuning {name_pr}...")
    best_pr_tuned = tune_best_model(best_pr, X_pr, y_pr, n_iter=20)

    artefacts["profit"] = {
        "model":            best_pr_tuned,
        "model_name":       name_pr,
        "features":         PROFIT_FEATURES,
        "results":          {k: {m: v for m, v in r.items() if m != "model"}
                             for k, r in results_pr.items()},
        "feature_importance": get_feature_importance(best_pr_tuned, PROFIT_FEATURES),
    }

    # ── Persist everything ──────────────────
    save_model(artefacts, "artefacts.pkl")
    save_model(df_enc,    "df_engineered.pkl")

    print("\n  ✓ All models trained and saved.")
    return artefacts
