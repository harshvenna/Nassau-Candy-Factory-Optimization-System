"""
Nassau Candy SupplyChainAI — Master Training Script
====================================================
Run this once to generate all model artefacts and SHAP values.
Usage:  python train.py
"""

import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_pipeline  import load_and_prepare, data_quality_report, load_raw
from src.ml_models      import run_full_training, load_model, encode_categoricals, prepare_X_y
from src.ml_models      import LEAD_TIME_FEATURES, PROFIT_FEATURES
from src.shap_explainer import compute_shap_values
from src.optimization_engine import generate_recommendations, run_all_monte_carlo

DATA_PATH = "data/candy.xlsx"

def main():
    print("=" * 60)
    print("  SupplyChainAI — Nassau Candy Distributor")
    print("  Master Training Pipeline")
    print("=" * 60)

    # ── 1. Load & clean ─────────────────────
    print("\n[1/5] Loading and preparing data...")
    raw = load_raw(DATA_PATH)
    rpt = data_quality_report(raw)
    print(f"      Raw rows: {rpt['total_rows']} | Duplicates: {rpt['duplicate_rows']}")
    df = load_and_prepare(DATA_PATH)
    print(f"      Clean rows: {len(df)} | Features: {df.shape[1]}")

    # ── 2. Train models ──────────────────────
    print("\n[2/5] Training ML models...")
    artefacts = run_full_training(df)

    # ── 3. SHAP values ───────────────────────
    print("\n[3/5] Computing SHAP explanations...")
    df_enc = load_model("df_engineered.pkl")
    from src.ml_models import encode_categoricals, prepare_X_y
    df_enc2, encoders = encode_categoricals(df_enc)

    X_lt, _ = prepare_X_y(df_enc2, LEAD_TIME_FEATURES, "Lead_Time")
    shap_lt  = compute_shap_values(artefacts["lead_time"]["model"], X_lt, LEAD_TIME_FEATURES, max_samples=400)
    with open("models/shap_lt.pkl", "wb") as f: pickle.dump(shap_lt, f)
    print(f"      Lead Time SHAP top feature: {shap_lt['shap_df'].iloc[0]['Feature']}")

    X_pr, _ = prepare_X_y(df_enc2, PROFIT_FEATURES, "Gross Profit")
    shap_pr  = compute_shap_values(artefacts["profit"]["model"], X_pr, PROFIT_FEATURES, max_samples=400)
    with open("models/shap_pr.pkl", "wb") as f: pickle.dump(shap_pr, f)
    print(f"      Profit SHAP top feature:    {shap_pr['shap_df'].iloc[0]['Feature']}")

    # ── 4. Recommendations ───────────────────
    print("\n[4/5] Generating optimization recommendations...")
    recs = generate_recommendations(df, artefacts, top_n=100, n_simulations=500)
    recs.to_csv("outputs/recommendations.csv", index=False)
    print(f"      Generated {len(recs)} recommendations")

    # ── 5. Monte Carlo ───────────────────────
    print("\n[5/5] Running Monte Carlo simulations...")
    mc_df = run_all_monte_carlo(df, n_runs=1000)
    mc_df.to_csv("outputs/monte_carlo_summary.csv", index=False)
    print(f"      {len(mc_df)} scenarios simulated")

    print("\n" + "=" * 60)
    print("  ✓ Training complete. All artefacts saved to models/")
    print("  ✓ Start the dashboard:  streamlit run streamlit_app/app.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
