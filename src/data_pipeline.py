"""
Nassau Candy SupplyChainAI — Data Pipeline
==========================================
Handles: loading, cleaning, feature engineering, factory mapping.
All downstream modules import from here.

Lead Time Strategy
------------------
The raw Ship Dates in the dataset are all recorded in 2026 while orders
are from 2024–2025, making direct subtraction meaningless for modelling.
Industry-standard logistics lead times are derived from Ship Mode and
factory-to-region shipping distance, with realistic Gaussian noise added
to create a learnable, non-trivial regression target. This matches
real-world supply-chain analytical methodology.
"""

import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────

FACTORY_COORDS = {
    "Lot's O' Nuts":     (32.881893, -111.768036),
    "Wicked Choccy's":   (32.076176,  -81.088371),
    "Sugar Shack":       (48.11914,   -96.18115),
    "Secret Factory":    (41.446333,  -90.565487),
    "The Other Factory": (35.1175,    -89.971107),
}

ALL_FACTORIES = list(FACTORY_COORDS.keys())

PRODUCT_FACTORY_MAP = {
    "Wonka Bar - Nutty Crunch Surprise":  "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows":          "Lot's O' Nuts",
    "Wonka Bar - Scrumdiddlyumptious":    "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate":         "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel":  "Wicked Choccy's",
    "Laffy Taffy":                        "Sugar Shack",
    "SweeTARTS":                          "Sugar Shack",
    "Nerds":                              "Sugar Shack",
    "Fun Dip":                            "Sugar Shack",
    "Fizzy Lifting Drinks":               "Sugar Shack",
    "Everlasting Gobstopper":             "Secret Factory",
    "Lickable Wallpaper":                 "Secret Factory",
    "Wonka Gum":                          "Secret Factory",
    "Hair Toffee":                        "The Other Factory",
    "Kazookles":                          "The Other Factory",
}

# Geographic centroids for each sales region
REGION_COORDS = {
    "Pacific":  (37.7749, -122.4194),
    "Interior": (41.8781,  -87.6298),
    "Atlantic": (40.7128,  -74.0060),
    "Gulf":     (29.7604,  -95.3698),
}

# Base lead-time days by ship mode
SHIP_MODE_BASE = {
    "Same Day":       1,
    "First Class":    3,
    "Second Class":   5,
    "Standard Class": 7,
}

# Division colour palette (for visualisations)
DIVISION_COLORS = {
    "Chocolate": "#7B3F00",
    "Sugar":     "#FF6B9D",
    "Other":     "#4A90D9",
}


# ─────────────────────────────────────────
# GEOGRAPHY HELPERS
# ─────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two coordinates."""
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def factory_to_region_km(factory: str, region: str) -> float:
    """Shipping distance from a factory to a region centroid (km)."""
    if factory not in FACTORY_COORDS or region not in REGION_COORDS:
        return np.nan
    flat, flon = FACTORY_COORDS[factory]
    rlat, rlon = REGION_COORDS[region]
    return haversine_km(flat, flon, rlat, rlon)


def compute_all_distances() -> pd.DataFrame:
    """
    Returns a DataFrame of distances (km) for every factory × region pair.
    Useful for geospatial visualisation.
    """
    rows = []
    for factory, (flat, flon) in FACTORY_COORDS.items():
        for region, (rlat, rlon) in REGION_COORDS.items():
            rows.append({
                "Factory": factory,
                "Region": region,
                "Distance_km": haversine_km(flat, flon, rlat, rlon),
                "Factory_Lat": flat,
                "Factory_Lon": flon,
                "Region_Lat": rlat,
                "Region_Lon": rlon,
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────
# LEAD TIME SIMULATION
# ─────────────────────────────────────────

def simulate_lead_time(
    ship_mode: str,
    factory: str,
    region: str,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Realistic lead time (days) = base_days + distance_factor + noise.

    distance_factor: 0–5 extra days scaled from max possible route (~5 000 km).
    noise: N(0, 0.8) truncated so output stays ≥ 1.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    base = SHIP_MODE_BASE.get(ship_mode, 5)
    dist = factory_to_region_km(factory, region)
    if np.isnan(dist):
        dist = 2000  # fallback
    dist_factor = (dist / 5000) * 5          # 0–5 extra days
    noise = rng.normal(0, 0.8)
    return max(1.0, round(base + dist_factor + noise, 1))


# ─────────────────────────────────────────
# LOADING & CLEANING
# ─────────────────────────────────────────

def load_raw(path: str = "data/candy.xlsx") -> pd.DataFrame:
    return pd.read_excel(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning pipeline:
    1. Standardise Order Date to datetime
    2. Normalise product name spelling variants
    3. Drop rows with missing critical fields
    4. Remove impossible numeric values (negative sales, etc.)
    5. Deduplicate on Order ID (keep first occurrence)
    """
    df = df.copy()

    # Dates — Order Date is reliable; Ship Date is unreliable so we ignore it
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")

    # Normalise product spelling
    df["Product Name"] = df["Product Name"].str.strip()
    SPELLING_FIX = {
        "Wonka Bar -Scrumdiddlyumptious": "Wonka Bar - Scrumdiddlyumptious",
        "Wonka Bar- Scrumdiddlyumptious": "Wonka Bar - Scrumdiddlyumptious",
    }
    df["Product Name"] = df["Product Name"].replace(SPELLING_FIX)

    # Drop rows missing essential fields
    df.dropna(subset=["Order Date", "Sales", "Cost", "Units", "Product Name", "Region"], inplace=True)

    # Remove bad numerics
    df = df[(df["Sales"] > 0) & (df["Cost"] >= 0) & (df["Units"] > 0)]

    # Deduplicate
    df.drop_duplicates(subset=["Order ID"], keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


# ─────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────

def engineer_features(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Adds all derived columns needed for ML models, optimisation engine,
    and Streamlit dashboard.
    """
    df = df.copy()
    rng = np.random.default_rng(seed)

    # ── Factory assignment ───────────────────
    df["Factory"] = df["Product Name"].map(PRODUCT_FACTORY_MAP)
    df["Factory_Lat"] = df["Factory"].map(
        lambda f: FACTORY_COORDS.get(f, (np.nan, np.nan))[0]
    )
    df["Factory_Lon"] = df["Factory"].map(
        lambda f: FACTORY_COORDS.get(f, (np.nan, np.nan))[1]
    )

    # ── Distance ─────────────────────────────
    df["Shipping_Distance_km"] = df.apply(
        lambda r: factory_to_region_km(r["Factory"], r["Region"]), axis=1
    )

    # ── Lead time (simulated from ship mode + distance + noise) ─
    lead_times = []
    for _, row in df.iterrows():
        lt = simulate_lead_time(row["Ship Mode"], row["Factory"], row["Region"], rng)
        lead_times.append(lt)
    df["Lead_Time"] = lead_times

    # ── Profitability metrics ────────────────
    df["Profit_Margin"]   = (df["Gross Profit"] / df["Sales"]).replace([np.inf, -np.inf], np.nan)
    df["Profit_Per_Unit"] = df["Gross Profit"] / df["Units"]
    df["Sales_Per_Unit"]  = df["Sales"] / df["Units"]
    df["Cost_Per_Unit"]   = df["Cost"] / df["Units"]
    df["Cost_Ratio"]      = (df["Cost"] / df["Sales"]).replace([np.inf, -np.inf], np.nan)

    # ── Time features ────────────────────────
    df["Order_Month"]     = df["Order Date"].dt.month
    df["Order_Quarter"]   = df["Order Date"].dt.quarter
    df["Order_Year"]      = df["Order Date"].dt.year
    df["Order_DayOfWeek"] = df["Order Date"].dt.dayofweek
    df["ShipMode_Days"]   = df["Ship Mode"].map(SHIP_MODE_BASE).fillna(5)

    # ── Aggregate demand features ────────────
    df["Region_Demand"]    = df.groupby("Region")["Units"].transform("sum")
    df["Customer_Value"]   = df.groupby("Customer ID")["Sales"].transform("sum")
    df["Demand_Score"]     = df.groupby(["Product Name", "Region"])["Units"].transform("sum")

    # ── Factory load ─────────────────────────
    factory_units          = df.groupby("Factory")["Units"].transform("sum")
    total_units            = df["Units"].sum()
    df["Factory_Load"]     = factory_units
    df["Factory_Load_Pct"] = (factory_units / total_units * 100).round(2)

    # ── Variability / stability ──────────────
    prod_mean = df.groupby("Product Name")["Gross Profit"].transform("mean")
    prod_std  = df.groupby("Product Name")["Gross Profit"].transform("std").fillna(0)
    df["Profit_Stability"]     = prod_mean / (prod_std + 1e-9)
    df["LeadTime_Variability"] = df.groupby(["Product Name", "Region"])["Lead_Time"].transform("std").fillna(0)

    # ── Route risk index ─────────────────────
    lt_max   = df["Lead_Time"].max()
    lt_min   = df["Lead_Time"].min()
    d_max    = df["Shipping_Distance_km"].max()
    d_min    = df["Shipping_Distance_km"].min()
    lt_norm  = (df["Lead_Time"]            - lt_min) / (lt_max - lt_min + 1e-9)
    d_norm   = (df["Shipping_Distance_km"] - d_min)  / (d_max - d_min + 1e-9)
    df["Route_Risk"] = (0.6 * lt_norm + 0.4 * d_norm).round(4)

    # ── Factory-level aggregates ─────────────
    df["Factory_Profitability"] = df.groupby("Factory")["Gross Profit"].transform("mean")
    df["Factory_Efficiency"]    = df.groupby("Factory")["Profit_Per_Unit"].transform("mean")

    # ── Product ranks ────────────────────────
    prod_profit_total          = df.groupby("Product Name")["Gross Profit"].transform("sum")
    prod_sales_total           = df.groupby("Product Name")["Sales"].transform("sum")
    df["Product_Profit_Rank"]  = prod_profit_total.rank(ascending=False, method="dense")
    df["Product_Sales_Rank"]   = prod_sales_total.rank(ascending=False, method="dense")

    # ── Final cleanup ────────────────────────
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    return df


# ─────────────────────────────────────────
# MASTER LOADER
# ─────────────────────────────────────────

def load_and_prepare(path: str = "data/candy.xlsx") -> pd.DataFrame:
    """One-call full pipeline: load → clean → feature engineer."""
    return engineer_features(clean(load_raw(path)))


# ─────────────────────────────────────────
# DATA QUALITY REPORT
# ─────────────────────────────────────────

def data_quality_report(df: pd.DataFrame) -> dict:
    """Returns structured audit metrics for the raw dataframe."""
    report = {
        "total_rows":     len(df),
        "total_columns":  len(df.columns),
        "duplicate_rows": int(df.duplicated(subset=["Order ID"]).sum()),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_pct":    (df.isnull().mean() * 100).round(2).to_dict(),
        "dtypes":         df.dtypes.astype(str).to_dict(),
    }
    outlier_counts = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        n_out = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
        outlier_counts[col] = n_out
    report["outlier_counts"] = outlier_counts
    return report
