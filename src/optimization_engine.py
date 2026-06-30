"""
Nassau Candy SupplyChainAI — Optimization & Recommendation Engine
=================================================================
For every product:
  1. Simulate reassignment to every factory
  2. Predict lead time and profit under each scenario
  3. Score each option using multi-objective optimization
  4. Generate ranked recommendations

Optimization Score = 0.40 × LeadTime_Reduction
                   + 0.30 × Profit_Improvement
                   + 0.20 × Risk_Reduction
                   + 0.10 × Confidence_Score

Also implements Monte Carlo simulation for uncertainty quantification.
"""

import numpy as np
import pandas as pd
from itertools import product as iterproduct
import warnings

warnings.filterwarnings("ignore")

from src.data_pipeline import (
    FACTORY_COORDS, ALL_FACTORIES, PRODUCT_FACTORY_MAP,
    REGION_COORDS, SHIP_MODE_BASE,
    factory_to_region_km, simulate_lead_time,
)

# Weights for multi-objective optimization score
SCORE_WEIGHTS = {
    "lead_time_reduction":  0.40,
    "profit_improvement":   0.30,
    "risk_reduction":       0.20,
    "confidence":           0.10,
}


# ─────────────────────────────────────────
# BASELINE METRICS PER PRODUCT × REGION
# ─────────────────────────────────────────

def compute_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per (Product, Region, Ship Mode)
    showing current factory performance metrics.
    """
    baseline = (
        df.groupby(["Product Name", "Region", "Ship Mode", "Factory"])
        .agg(
            Avg_Lead_Time   = ("Lead_Time",    "mean"),
            Std_Lead_Time   = ("Lead_Time",    "std"),
            Avg_Profit      = ("Gross Profit", "mean"),
            Avg_Margin      = ("Profit_Margin","mean"),
            Total_Units     = ("Units",        "sum"),
            Avg_Distance    = ("Shipping_Distance_km", "mean"),
            Avg_Route_Risk  = ("Route_Risk",   "mean"),
            Order_Count     = ("Order ID",     "count"),
        )
        .reset_index()
    )
    baseline["Std_Lead_Time"] = baseline["Std_Lead_Time"].fillna(0)
    return baseline


# ─────────────────────────────────────────
# FACTORY SIMULATION
# ─────────────────────────────────────────

def simulate_factory_assignment(
    product: str,
    region: str,
    ship_mode: str,
    alt_factory: str,
    lead_time_model,
    profit_model,
    encoders: dict,
    df: pd.DataFrame,
    feature_cols_lt: list,
    feature_cols_pr: list,
    n_simulations: int = 200,
    seed: int = 42,
) -> dict:
    """
    Predicts lead time and profit if 'product' were manufactured at 'alt_factory'
    instead of its current factory, and shipped to 'region' via 'ship_mode'.

    Returns a dict with predicted metrics.
    """
    rng = np.random.default_rng(seed)

    dist_km = factory_to_region_km(alt_factory, region)
    if np.isnan(dist_km):
        dist_km = 2000.0

    base_days = SHIP_MODE_BASE.get(ship_mode, 5)

    # Simulate lead times with noise to get a distribution
    lt_samples = np.array([
        simulate_lead_time(ship_mode, alt_factory, region, rng)
        for _ in range(n_simulations)
    ])
    predicted_lt_mean = lt_samples.mean()
    predicted_lt_std  = lt_samples.std()
    predicted_lt_ci95 = (
        np.percentile(lt_samples, 2.5),
        np.percentile(lt_samples, 97.5),
    )

    # Derive a risk score for this factory-region pair
    # Higher distance and higher lead time = higher risk
    max_dist = max(factory_to_region_km(f, r) for f in ALL_FACTORIES for r in REGION_COORDS)
    dist_norm = dist_km / max_dist
    lt_norm   = predicted_lt_mean / 13.0   # max possible ~13 days
    risk_score = 0.6 * lt_norm + 0.4 * dist_norm

    # Use product median profit from data as profit proxy
    # (a full feature vector prediction would require matching all fields)
    product_df = df[df["Product Name"] == product]
    if len(product_df) > 0:
        base_profit = product_df["Gross Profit"].median()
        # Slight profit penalty for longer routes (logistics cost proxy)
        profit_adjustment = 1 - (dist_km / max_dist) * 0.08
        predicted_profit = base_profit * profit_adjustment
    else:
        predicted_profit = 5.0

    # Confidence: based on data availability and simulation convergence
    cv = predicted_lt_std / (predicted_lt_mean + 1e-9)
    confidence = float(np.clip(1 - cv * 2, 0.3, 1.0))

    return {
        "Alt_Factory":         alt_factory,
        "Alt_Dist_km":         dist_km,
        "Pred_Lead_Time_Mean": round(predicted_lt_mean, 2),
        "Pred_Lead_Time_Std":  round(predicted_lt_std, 2),
        "Pred_Lead_Time_CI95": (round(predicted_lt_ci95[0], 2), round(predicted_lt_ci95[1], 2)),
        "Pred_Profit":         round(predicted_profit, 2),
        "Risk_Score":          round(risk_score, 4),
        "Confidence":          round(confidence, 4),
    }


# ─────────────────────────────────────────
# MULTI-OBJECTIVE OPTIMIZATION SCORE
# ─────────────────────────────────────────

def compute_optimization_score(
    current_lt: float,
    alt_lt: float,
    current_profit: float,
    alt_profit: float,
    current_risk: float,
    alt_risk: float,
    confidence: float,
    weights: dict = SCORE_WEIGHTS,
) -> dict:
    """
    Returns individual component scores and composite optimization score (0–100).
    """
    # Lead time reduction (positive = improvement)
    if current_lt > 0:
        lt_red = (current_lt - alt_lt) / current_lt
    else:
        lt_red = 0.0

    # Profit improvement
    if current_profit > 0:
        pr_imp = (alt_profit - current_profit) / current_profit
    else:
        pr_imp = 0.0

    # Risk reduction
    if current_risk > 0:
        risk_red = (current_risk - alt_risk) / current_risk
    else:
        risk_red = 0.0

    # Composite score (normalised to 0–100)
    raw = (
        weights["lead_time_reduction"] * np.clip(lt_red,  -1, 1) +
        weights["profit_improvement"]  * np.clip(pr_imp,  -1, 1) +
        weights["risk_reduction"]      * np.clip(risk_red,-1, 1) +
        weights["confidence"]          * confidence
    )
    score = round((raw + 1) / 2 * 100, 2)   # map [-1,1] → [0,100]

    return {
        "LT_Reduction_Pct":   round(lt_red  * 100, 2),
        "Profit_Improve_Pct": round(pr_imp  * 100, 2),
        "Risk_Reduction_Pct": round(risk_red * 100, 2),
        "Confidence_Score":   round(confidence * 100, 2),
        "Optimization_Score": score,
    }


# ─────────────────────────────────────────
# FULL RECOMMENDATION GENERATION
# ─────────────────────────────────────────

def generate_recommendations(
    df: pd.DataFrame,
    artefacts: dict,
    top_n: int = 10,
    n_simulations: int = 200,
) -> pd.DataFrame:
    """
    For every unique (Product, Region, Ship Mode) combination,
    simulates all possible factory reassignments and generates
    ranked recommendations.

    Returns a DataFrame of top-N recommendations sorted by Optimization_Score.
    """
    lt_model  = artefacts["lead_time"]["model"]
    pr_model  = artefacts["profit"]["model"]
    encoders  = artefacts["encoders"]
    feat_lt   = artefacts["lead_time"]["features"]
    feat_pr   = artefacts["profit"]["features"]

    baseline = compute_baseline(df)
    recommendations = []

    combos = df[["Product Name", "Region", "Ship Mode"]].drop_duplicates().values

    for (product, region, ship_mode) in combos:
        current_factory = PRODUCT_FACTORY_MAP.get(product, "Unknown")

        # Current baseline metrics
        cur_row = baseline[
            (baseline["Product Name"] == product) &
            (baseline["Region"]       == region) &
            (baseline["Ship Mode"]    == ship_mode)
        ]
        if cur_row.empty:
            continue

        cur_lt    = float(cur_row["Avg_Lead_Time"].values[0])
        cur_risk  = float(cur_row["Avg_Route_Risk"].values[0])
        cur_prof  = float(cur_row["Avg_Profit"].values[0])
        cur_dist  = float(cur_row["Avg_Distance"].values[0])

        # Simulate every alternative factory
        for alt_factory in ALL_FACTORIES:
            if alt_factory == current_factory:
                continue

            sim = simulate_factory_assignment(
                product, region, ship_mode, alt_factory,
                lt_model, pr_model, encoders, df,
                feat_lt, feat_pr,
                n_simulations=n_simulations,
            )

            scores = compute_optimization_score(
                current_lt=cur_lt,
                alt_lt=sim["Pred_Lead_Time_Mean"],
                current_profit=cur_prof,
                alt_profit=sim["Pred_Profit"],
                current_risk=cur_risk,
                alt_risk=sim["Risk_Score"],
                confidence=sim["Confidence"],
            )

            recommendations.append({
                "Product":                product,
                "Region":                 region,
                "Ship_Mode":              ship_mode,
                "Current_Factory":        current_factory,
                "Recommended_Factory":    alt_factory,
                "Current_Distance_km":    round(cur_dist, 0),
                "Alt_Distance_km":        round(sim["Alt_Dist_km"], 0),
                "Current_Lead_Time_Days": round(cur_lt, 2),
                "Pred_Lead_Time_Days":    sim["Pred_Lead_Time_Mean"],
                "Pred_Lead_Time_CI_Low":  sim["Pred_Lead_Time_CI95"][0],
                "Pred_Lead_Time_CI_High": sim["Pred_Lead_Time_CI95"][1],
                "Current_Profit":         round(cur_prof, 2),
                "Pred_Profit":            sim["Pred_Profit"],
                "Current_Risk":           round(cur_risk, 4),
                "Alt_Risk":               sim["Risk_Score"],
                **scores,
            })

    rec_df = pd.DataFrame(recommendations)
    if rec_df.empty:
        return rec_df

    rec_df = rec_df.sort_values("Optimization_Score", ascending=False).reset_index(drop=True)
    return rec_df.head(top_n) if top_n else rec_df


# ─────────────────────────────────────────
# MONTE CARLO SIMULATION
# ─────────────────────────────────────────

def monte_carlo_simulation(
    product: str,
    region: str,
    ship_mode: str,
    factory: str,
    n_runs: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Runs n_runs simulations of lead time for a given scenario.
    Returns statistical summary including confidence intervals.
    """
    rng = np.random.default_rng(seed)
    samples = np.array([
        simulate_lead_time(ship_mode, factory, region, rng)
        for _ in range(n_runs)
    ])

    return {
        "product":      product,
        "region":       region,
        "ship_mode":    ship_mode,
        "factory":      factory,
        "n_runs":       n_runs,
        "mean":         round(samples.mean(), 3),
        "std":          round(samples.std(), 3),
        "median":       round(np.median(samples), 3),
        "p5":           round(np.percentile(samples, 5), 3),
        "p25":          round(np.percentile(samples, 25), 3),
        "p75":          round(np.percentile(samples, 75), 3),
        "p95":          round(np.percentile(samples, 95), 3),
        "best_case":    round(samples.min(), 3),
        "worst_case":   round(samples.max(), 3),
        "ci_95_low":    round(np.percentile(samples, 2.5), 3),
        "ci_95_high":   round(np.percentile(samples, 97.5), 3),
        "samples":      samples,   # kept for plotting distributions
    }


def run_all_monte_carlo(
    df: pd.DataFrame,
    n_runs: int = 1000,
) -> pd.DataFrame:
    """
    Runs Monte Carlo for all unique (Product, Region, Ship Mode, Factory) combos.
    Returns summary DataFrame.
    """
    combos = (
        df[["Product Name", "Region", "Ship Mode", "Factory"]]
        .drop_duplicates()
        .values
    )
    rows = []
    for (product, region, ship_mode, factory) in combos:
        mc = monte_carlo_simulation(product, region, ship_mode, factory, n_runs)
        rows.append({k: v for k, v in mc.items() if k != "samples"})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────
# SCENARIO COMPARISON (what-if)
# ─────────────────────────────────────────

def compare_scenarios(
    product: str,
    region: str,
    ship_mode: str,
    factory_a: str,
    factory_b: str,
    n_runs: int = 1000,
) -> dict:
    """
    Side-by-side Monte Carlo comparison of two factory assignments.
    Used by the Streamlit What-If module.
    """
    mc_a = monte_carlo_simulation(product, region, ship_mode, factory_a, n_runs)
    mc_b = monte_carlo_simulation(product, region, ship_mode, factory_b, n_runs)

    delta_mean = mc_a["mean"] - mc_b["mean"]
    pct_change = (delta_mean / mc_a["mean"] * 100) if mc_a["mean"] != 0 else 0

    return {
        "scenario_a":       mc_a,
        "scenario_b":       mc_b,
        "delta_mean_days":  round(delta_mean, 3),
        "improvement_pct":  round(pct_change, 2),
        "recommendation":   factory_b if delta_mean > 0 else factory_a,
    }
