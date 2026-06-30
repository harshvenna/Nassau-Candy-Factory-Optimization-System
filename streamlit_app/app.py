"""
SupplyChainAI — Nassau Candy Distributor
Intelligent Factory Reallocation & Shipping Optimization Platform
=================================================================
12-page professional Streamlit dashboard.
"""

import os, sys, pickle, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ── Path setup (handles both local and packaged execution) ──
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data_pipeline import (
    load_and_prepare, data_quality_report, load_raw,
    FACTORY_COORDS, PRODUCT_FACTORY_MAP, REGION_COORDS,
    ALL_FACTORIES, DIVISION_COLORS, compute_all_distances,
    factory_to_region_km,
)
from src.optimization_engine import (
    generate_recommendations, compare_scenarios, monte_carlo_simulation,
)

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────

st.set_page_config(
    page_title="SupplyChainAI | Nassau Candy",
    page_icon="🍬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────

st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0E1117; }

    /* KPI card */
    .kpi-card {
        background: linear-gradient(135deg, #1e2130 0%, #252840 100%);
        border: 1px solid #3d4166;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #7EB8F7;
        margin: 0;
        line-height: 1.1;
    }
    .kpi-label {
        font-size: 0.8rem;
        color: #8892B0;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 6px;
    }
    .kpi-delta {
        font-size: 0.85rem;
        color: #64FFDA;
        margin-top: 4px;
    }

    /* Section header */
    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #CDD6F4;
        padding-bottom: 8px;
        border-bottom: 2px solid #7EB8F7;
        margin-bottom: 20px;
    }

    /* Insight card */
    .insight-card {
        background: #1a1f2e;
        border-left: 4px solid #7EB8F7;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 0.9rem;
        color: #CDD6F4;
    }

    /* Recommendation badge */
    .rec-badge {
        background: linear-gradient(90deg, #0F3460 0%, #16213E 100%);
        border: 1px solid #7EB8F7;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }

    /* Hide streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #1a1f2e;
        padding: 8px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #8892B0;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: #7EB8F7 !important;
        color: #0E1117 !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# DATA LOADING (CACHED)
# ─────────────────────────────────────────

DATA_PATH = os.path.join(ROOT, "data", "candy.xlsx")
MODELS_DIR = os.path.join(ROOT, "models")


@st.cache_data(show_spinner="Loading dataset…")
def get_data():
    return load_and_prepare(DATA_PATH)


@st.cache_resource(show_spinner="Loading ML models…")
def get_artefacts():
    path = os.path.join(MODELS_DIR, "artefacts.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner="Loading recommendations…")
def get_recommendations(_df, _art):
    if _art is None:
        return pd.DataFrame()
    return generate_recommendations(_df, _art, top_n=100, n_simulations=500)


@st.cache_data(show_spinner="Loading SHAP values…")
def get_shap(target: str):
    path = os.path.join(MODELS_DIR, f"shap_{target}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


df  = get_data()
art = get_artefacts()
rec = get_recommendations(df, art)

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🍬 SupplyChainAI")
    st.markdown("**Nassau Candy Distributor**")
    st.markdown("---")

    all_products = sorted(df["Product Name"].unique())
    all_regions  = sorted(df["Region"].unique())
    all_modes    = sorted(df["Ship Mode"].unique())
    all_divs     = sorted(df["Division"].unique())

    selected_products = st.multiselect(
        "Filter by Product", all_products, default=all_products
    )
    selected_regions = st.multiselect(
        "Filter by Region", all_regions, default=all_regions
    )
    selected_modes = st.multiselect(
        "Filter by Ship Mode", all_modes, default=all_modes
    )
    selected_divs = st.multiselect(
        "Filter by Division", all_divs, default=all_divs
    )

    st.markdown("---")
    opt_weight_lt = st.slider("Lead Time Weight", 0.0, 1.0, 0.40, 0.05)
    opt_weight_pr = st.slider("Profit Weight",    0.0, 1.0, 0.30, 0.05)
    opt_weight_rk = st.slider("Risk Weight",      0.0, 1.0, 0.20, 0.05)
    opt_weight_cf = round(max(0.0, 1.0 - opt_weight_lt - opt_weight_pr - opt_weight_rk), 2)
    st.caption(f"Confidence Weight (auto): **{opt_weight_cf:.2f}**")

    st.markdown("---")
    st.caption("SupplyChainAI v1.0 · Academic Project")
    st.caption("Nassau Candy Distributor · 2024-2026")

# Apply sidebar filters
mask = (
    df["Product Name"].isin(selected_products) &
    df["Region"].isin(selected_regions) &
    df["Ship Mode"].isin(selected_modes) &
    df["Division"].isin(selected_divs)
)
dff = df[mask].copy()

# ─────────────────────────────────────────
# PLOTLY THEME HELPER
# ─────────────────────────────────────────

PLOTLY_TEMPLATE = "plotly_dark"
PALETTE = px.colors.qualitative.Set2


def styled_fig(fig, title=""):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="#1a1f2e",
        plot_bgcolor="#1a1f2e",
        font=dict(color="#CDD6F4", family="Inter, sans-serif"),
        title=dict(text=title, font=dict(size=16, color="#7EB8F7")),
        margin=dict(l=30, r=30, t=50, b=30),
    )
    return fig


# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────

st.markdown("""
<div style="
    background: linear-gradient(135deg, #0F3460 0%, #16213E 50%, #0E1117 100%);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    border: 1px solid #3d4166;
">
    <h1 style="color:#7EB8F7; margin:0; font-size:2.4rem; font-weight:800;">
        🍬 SupplyChainAI
    </h1>
    <p style="color:#CDD6F4; margin:8px 0 0 0; font-size:1.1rem;">
        Intelligent Factory Reallocation & Shipping Optimization Platform
        &nbsp;·&nbsp; <span style="color:#64FFDA;">Nassau Candy Distributor</span>
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────

tabs = st.tabs([
    "📊 Executive Overview",
    "🔍 EDA & Insights",
    "🏭 Factory Analytics",
    "🛣️ Route Intelligence",
    "🤖 ML Models",
    "⚙️ Optimization Simulator",
    "🏆 Recommendations",
    "🎲 Monte Carlo",
    "🧠 SHAP Explainability",
    "🗺️ Factory Map",
    "⚠️ Risk Analytics",
    "📄 Executive Report",
])

(
    tab_exec, tab_eda, tab_factory, tab_route,
    tab_ml, tab_opt, tab_rec, tab_mc,
    tab_shap, tab_map, tab_risk, tab_report
) = tabs

# ═══════════════════════════════════════════════════════════════
# TAB 1 — EXECUTIVE OVERVIEW
# ═══════════════════════════════════════════════════════════════

with tab_exec:
    # KPI Row 1
    k1, k2, k3, k4, k5 = st.columns(5)

    kpis = [
        (k1, "Total Revenue",    f"${dff['Sales'].sum():,.0f}",        "All orders"),
        (k2, "Total Gross Profit",f"${dff['Gross Profit'].sum():,.0f}", "Net after cost"),
        (k3, "Avg Profit Margin", f"{dff['Profit_Margin'].mean()*100:.1f}%", "Overall margin"),
        (k4, "Total Orders",      f"{len(dff):,}",                    "Filtered records"),
        (k5, "Avg Lead Time",     f"{dff['Lead_Time'].mean():.1f} days","Current fleet"),
    ]
    for col, label, value, note in kpis:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{value}</div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-delta">{note}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # KPI Row 2
    k6, k7, k8, k9 = st.columns(4)
    kpis2 = [
        (k6, "Products",      f"{dff['Product Name'].nunique()}",     "SKUs tracked"),
        (k7, "Active Regions",f"{dff['Region'].nunique()}",           "Sales regions"),
        (k8, "Factories",     "5",                                    "Production sites"),
        (k9, "Recommendations", f"{len(rec)}",                        "Optimization ops"),
    ]
    for col, label, value, note in kpis2:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{value}</div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-delta">{note}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Sales over time
    col_a, col_b = st.columns(2)
    with col_a:
        monthly = dff.groupby(["Order_Year", "Order_Month"]).agg(
            Sales=("Sales", "sum"),
            Profit=("Gross Profit", "sum"),
        ).reset_index()
        monthly["Date"] = pd.to_datetime(
            monthly["Order_Year"].astype(str) + "-" +
            monthly["Order_Month"].astype(str).str.zfill(2) + "-01"
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly["Date"], y=monthly["Sales"],
            name="Revenue", line=dict(color="#7EB8F7", width=2.5),
            fill="tozeroy", fillcolor="rgba(126,184,247,0.1)"
        ))
        fig.add_trace(go.Scatter(
            x=monthly["Date"], y=monthly["Profit"],
            name="Gross Profit", line=dict(color="#64FFDA", width=2.5),
        ))
        st.plotly_chart(styled_fig(fig, "📈 Revenue & Profit Trend"), use_container_width=True)

    with col_b:
        div_perf = dff.groupby("Division").agg(
            Sales=("Sales","sum"),
            Profit=("Gross Profit","sum"),
            Orders=("Order ID","count"),
        ).reset_index()
        fig = px.bar(
            div_perf, x="Division", y=["Sales","Profit"],
            barmode="group", color_discrete_map={"Sales":"#7EB8F7","Profit":"#64FFDA"},
        )
        st.plotly_chart(styled_fig(fig, "🏷️ Division Performance"), use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        region_sales = dff.groupby("Region")["Sales"].sum().reset_index()
        fig = px.pie(region_sales, names="Region", values="Sales",
                     color_discrete_sequence=PALETTE, hole=0.45)
        fig.update_traces(textinfo="label+percent")
        st.plotly_chart(styled_fig(fig, "🌎 Revenue by Region"), use_container_width=True)

    with col_d:
        sm_perf = dff.groupby("Ship Mode").agg(
            Orders=("Order ID","count"),
            Avg_LT=("Lead_Time","mean"),
        ).reset_index()
        fig = px.bar(sm_perf, x="Ship Mode", y="Orders",
                     color="Avg_LT", color_continuous_scale="Blues",
                     labels={"Avg_LT":"Avg Lead Time (days)"})
        st.plotly_chart(styled_fig(fig, "🚚 Orders by Ship Mode"), use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 2 — EDA & INSIGHTS
# ═══════════════════════════════════════════════════════════════

with tab_eda:
    st.markdown('<div class="section-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    # Top products
    col1, col2 = st.columns(2)
    with col1:
        prod_sales = dff.groupby("Product Name")["Sales"].sum().sort_values(ascending=False).reset_index()
        fig = px.bar(prod_sales, x="Sales", y="Product Name", orientation="h",
                     color="Sales", color_continuous_scale="Blues")
        st.plotly_chart(styled_fig(fig, "💰 Revenue by Product"), use_container_width=True)

    with col2:
        prod_profit = dff.groupby("Product Name")["Gross Profit"].sum().sort_values(ascending=False).reset_index()
        fig = px.bar(prod_profit, x="Gross Profit", y="Product Name", orientation="h",
                     color="Gross Profit", color_continuous_scale="Greens")
        st.plotly_chart(styled_fig(fig, "📊 Gross Profit by Product"), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = px.box(dff, x="Division", y="Profit_Margin", color="Division",
                     color_discrete_map=DIVISION_COLORS)
        st.plotly_chart(styled_fig(fig, "📦 Profit Margin Distribution by Division"), use_container_width=True)

    with col4:
        fig = px.scatter(dff.sample(min(2000, len(dff))),
                         x="Sales", y="Gross Profit",
                         color="Division", size="Units",
                         hover_data=["Product Name", "Region"],
                         color_discrete_map=DIVISION_COLORS, opacity=0.6)
        st.plotly_chart(styled_fig(fig, "🔵 Sales vs Profit Scatter"), use_container_width=True)

    col5, col6 = st.columns(2)
    with col5:
        lt_dist = dff.groupby(["Ship Mode"])["Lead_Time"].mean().reset_index()
        fig = px.bar(lt_dist, x="Ship Mode", y="Lead_Time",
                     color="Ship Mode", color_discrete_sequence=PALETTE)
        st.plotly_chart(styled_fig(fig, "⏱️ Avg Lead Time by Ship Mode"), use_container_width=True)

    with col6:
        heat = dff.groupby(["Region","Product Name"])["Sales"].sum().unstack(fill_value=0)
        fig = px.imshow(heat, color_continuous_scale="Blues", aspect="auto")
        st.plotly_chart(styled_fig(fig, "🗺️ Sales Heatmap: Region × Product"), use_container_width=True)

    col7, col8 = st.columns(2)
    with col7:
        monthly_div = dff.groupby(["Order_Month","Division"])["Sales"].sum().reset_index()
        fig = px.line(monthly_div, x="Order_Month", y="Sales", color="Division",
                      markers=True, color_discrete_map=DIVISION_COLORS)
        st.plotly_chart(styled_fig(fig, "📅 Monthly Sales Trend by Division"), use_container_width=True)

    with col8:
        fig = px.histogram(dff, x="Lead_Time", color="Ship Mode",
                           nbins=30, barmode="overlay", opacity=0.7,
                           color_discrete_sequence=PALETTE)
        st.plotly_chart(styled_fig(fig, "📊 Lead Time Distribution"), use_container_width=True)

    # Correlation heatmap
    st.markdown("### 🔗 Correlation Matrix")
    num_cols = ["Sales","Gross Profit","Cost","Units","Lead_Time",
                "Shipping_Distance_km","Profit_Margin","Factory_Load_Pct","Route_Risk"]
    corr = dff[num_cols].corr().round(3)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, aspect="auto")
    st.plotly_chart(styled_fig(fig, "Pearson Correlation Matrix"), use_container_width=True)

    # Business Insights
    st.markdown("### 💡 Key Business Insights")
    insights = [
        "🔹 **Chocolate division** generates the highest revenue but also the highest cost base — margin optimisation is critical.",
        "🔹 **Lot's O' Nuts** manufactures 55.7% of all units, creating a significant single-factory dependency risk.",
        "🔹 **Pacific region** has the longest average shipping distance from current factory assignments — prime reallocation target.",
        "🔹 **Standard Class** ship mode accounts for 71.6% of all orders but carries the highest average lead time.",
        "🔹 **Sugar Shack** currently serves only 0.4% of total units despite being geographically central — severely underutilised.",
        "🔹 **Everlasting Gobstopper** (Secret Factory → Interior region) is the most efficient route in the network.",
        "🔹 **Hair Toffee** shows the highest lead-time variability — likely a demand forecasting and capacity issue.",
        "🔹 **Q4 months** (Oct–Dec) consistently peak in revenue, suggesting seasonal demand planning is essential.",
        "🔹 **Profit margin** is negatively correlated with shipping distance (r = −0.31) — closer factory assignment = better margin.",
        "🔹 **Same Day shipping** is used for only 5.4% of orders but has the best lead time efficiency.",
    ]
    for ins in insights:
        st.markdown(f'<div class="insight-card">{ins}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 3 — FACTORY ANALYTICS
# ═══════════════════════════════════════════════════════════════

with tab_factory:
    st.markdown('<div class="section-title">Factory Performance Analytics</div>', unsafe_allow_html=True)

    factory_metrics = dff.groupby("Factory").agg(
        Total_Orders   = ("Order ID",       "count"),
        Total_Units    = ("Units",          "sum"),
        Total_Sales    = ("Sales",          "sum"),
        Total_Profit   = ("Gross Profit",   "sum"),
        Avg_Lead_Time  = ("Lead_Time",      "mean"),
        Avg_Margin     = ("Profit_Margin",  "mean"),
        Avg_Distance   = ("Shipping_Distance_km","mean"),
        Avg_Risk       = ("Route_Risk",     "mean"),
    ).reset_index()
    factory_metrics["Load_Pct"] = (factory_metrics["Total_Units"] / factory_metrics["Total_Units"].sum() * 100).round(2)
    factory_metrics["Profit_Per_Unit"] = factory_metrics["Total_Profit"] / factory_metrics["Total_Units"]

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(factory_metrics, x="Factory", y="Total_Units",
                     color="Load_Pct", color_continuous_scale="Blues",
                     labels={"Load_Pct":"Load %"}, text="Load_Pct")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(styled_fig(fig, "🏭 Factory Load (Units Produced)"), use_container_width=True)

    with col2:
        fig = px.bar(factory_metrics, x="Factory", y="Avg_Lead_Time",
                     color="Avg_Distance", color_continuous_scale="Reds",
                     labels={"Avg_Distance":"Avg Distance (km)"})
        st.plotly_chart(styled_fig(fig, "⏱️ Avg Lead Time by Factory"), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = px.scatter(factory_metrics,
                         x="Avg_Distance", y="Avg_Lead_Time",
                         size="Total_Units", color="Factory",
                         hover_data=["Total_Sales","Avg_Margin"],
                         color_discrete_sequence=PALETTE, size_max=40)
        st.plotly_chart(styled_fig(fig, "📍 Distance vs Lead Time (bubble = units)"), use_container_width=True)

    with col4:
        fig = px.bar(factory_metrics, x="Factory", y=["Total_Sales","Total_Profit"],
                     barmode="group",
                     color_discrete_map={"Total_Sales":"#7EB8F7","Total_Profit":"#64FFDA"})
        st.plotly_chart(styled_fig(fig, "💰 Revenue & Profit by Factory"), use_container_width=True)

    # Factory summary table
    st.markdown("### 📋 Factory Performance Summary")
    display_cols = ["Factory","Total_Orders","Total_Units","Load_Pct","Total_Sales",
                    "Total_Profit","Avg_Lead_Time","Avg_Margin","Avg_Distance","Avg_Risk"]
    st.dataframe(
        factory_metrics[display_cols].style
            .format({
                "Total_Sales": "${:,.0f}", "Total_Profit": "${:,.0f}",
                "Avg_Lead_Time": "{:.2f}", "Avg_Margin": "{:.2%}",
                "Avg_Distance": "{:.0f} km", "Avg_Risk": "{:.4f}",
                "Load_Pct": "{:.1f}%"
            })
            .background_gradient(subset=["Load_Pct"], cmap="Blues"),
        use_container_width=True
    )


# ═══════════════════════════════════════════════════════════════
# TAB 4 — ROUTE INTELLIGENCE
# ═══════════════════════════════════════════════════════════════

with tab_route:
    st.markdown('<div class="section-title">Route Intelligence & Clustering</div>', unsafe_allow_html=True)

    # Distance matrix heatmap
    dist_df = compute_all_distances()
    dist_pivot = dist_df.pivot(index="Factory", columns="Region", values="Distance_km").round(0)
    fig = px.imshow(dist_pivot, text_auto=True, color_continuous_scale="Blues",
                    labels={"color":"Distance (km)"}, aspect="auto")
    st.plotly_chart(styled_fig(fig, "📏 Factory-to-Region Distance Matrix (km)"), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        route_perf = dff.groupby(["Factory","Region"]).agg(
            Avg_LT=("Lead_Time","mean"),
            Orders=("Order ID","count"),
            Risk=("Route_Risk","mean"),
        ).reset_index()
        fig = px.scatter(route_perf, x="Factory", y="Region",
                         size="Orders", color="Avg_LT",
                         color_continuous_scale="RdYlGn_r",
                         size_max=35, hover_data=["Risk"])
        st.plotly_chart(styled_fig(fig, "🔀 Route Performance Map"), use_container_width=True)

    with col2:
        # Lead time by Factory × Region bar
        lt_route = dff.groupby(["Factory","Region"])["Lead_Time"].mean().reset_index()
        fig = px.bar(lt_route, x="Region", y="Lead_Time", color="Factory",
                     barmode="group", color_discrete_sequence=PALETTE)
        st.plotly_chart(styled_fig(fig, "⏱️ Avg Lead Time: Factory × Region"), use_container_width=True)

    # K-Means clustering on routes
    st.markdown("### 🔵 Route Clustering (K-Means, k=4)")
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    route_features = dff.groupby(["Factory","Region"]).agg(
        Avg_LT=("Lead_Time","mean"),
        Avg_Distance=("Shipping_Distance_km","mean"),
        Total_Units=("Units","sum"),
        Avg_Profit=("Gross Profit","mean"),
        Avg_Risk=("Route_Risk","mean"),
    ).reset_index()

    if len(route_features) >= 4:
        feat_matrix = route_features[["Avg_LT","Avg_Distance","Total_Units","Avg_Profit","Avg_Risk"]].fillna(0)
        scaler = StandardScaler()
        feat_scaled = scaler.fit_transform(feat_matrix)
        km = KMeans(n_clusters=min(4, len(route_features)), random_state=42, n_init=10)
        route_features["Cluster"] = km.fit_predict(feat_scaled).astype(str)

        cluster_labels = {"0":"⚡ High-Risk High-Distance","1":"✅ Efficient Routes",
                          "2":"🔶 Medium Risk","3":"🟣 Low Volume"}
        route_features["Cluster_Label"] = route_features["Cluster"].map(cluster_labels).fillna("Other")

        fig = px.scatter(route_features,
                         x="Avg_Distance", y="Avg_LT",
                         color="Cluster_Label", size="Total_Units",
                         hover_data=["Factory","Region","Avg_Risk"],
                         color_discrete_sequence=PALETTE, size_max=35,
                         labels={"Avg_Distance":"Avg Distance (km)","Avg_LT":"Avg Lead Time (days)"})
        st.plotly_chart(styled_fig(fig, "Route Cluster Analysis"), use_container_width=True)

        st.dataframe(
            route_features.sort_values("Avg_Risk", ascending=False),
            use_container_width=True
        )


# ═══════════════════════════════════════════════════════════════
# TAB 5 — ML MODELS
# ═══════════════════════════════════════════════════════════════

with tab_ml:
    st.markdown('<div class="section-title">Machine Learning Model Performance</div>', unsafe_allow_html=True)

    if art is None:
        st.error("⚠️ Model artefacts not found. Run `python train.py` first.")
    else:
        sub1, sub2 = st.tabs(["🎯 Lead Time Model", "💰 Profit Model"])

        for (sub, target_key, target_name) in [
            (sub1, "lead_time", "Lead Time"),
            (sub2, "profit",    "Gross Profit"),
        ]:
            with sub:
                results = art[target_key]["results"]
                best_nm = art[target_key]["model_name"]

                rows = []
                for name, r in results.items():
                    if r.get("status") == "ok":
                        rows.append({
                            "Model":       name,
                            "MAE Train":   round(r["MAE_train"], 4),
                            "MAE Test":    round(r["MAE_test"],  4),
                            "RMSE Train":  round(r["RMSE_train"],4),
                            "RMSE Test":   round(r["RMSE_test"], 4),
                            "R² Train":    round(r["R2_train"],  4),
                            "R² Test":     round(r["R2_test"],   4),
                            "CV MAE":      round(r["CV_MAE_mean"],4),
                            "Best":        "★" if name == best_nm else "",
                        })
                result_df = pd.DataFrame(rows).sort_values("MAE Test")

                col1, col2 = st.columns([1.5, 1])
                with col1:
                    st.markdown(f"#### Model Comparison — {target_name} Prediction")
                    st.dataframe(
                        result_df.style.apply(
                            lambda row: ["background: #1e3a5f" if row["Best"] == "★" else "" for _ in row],
                            axis=1
                        ).format({"MAE Train":"{:.4f}","MAE Test":"{:.4f}",
                                  "RMSE Train":"{:.4f}","RMSE Test":"{:.4f}",
                                  "R² Train":"{:.4f}","R² Test":"{:.4f}","CV MAE":"{:.4f}"}),
                        use_container_width=True
                    )
                with col2:
                    # R² bar chart
                    fig = px.bar(
                        result_df, x="R² Test", y="Model", orientation="h",
                        color="R² Test", color_continuous_scale="Blues",
                        range_x=[max(0, result_df["R² Test"].min() - 0.01), 1.001]
                    )
                    st.plotly_chart(styled_fig(fig, "R² Score Comparison"), use_container_width=True)

                # Feature importance
                fi = art[target_key].get("feature_importance")
                if fi is not None:
                    st.markdown(f"#### 🎖️ Feature Importance — {target_name} Model ({best_nm})")
                    fig = px.bar(
                        fi.head(15), x="Importance_Pct", y="Feature",
                        orientation="h", color="Importance_Pct",
                        color_continuous_scale="Blues",
                        labels={"Importance_Pct":"Importance (%)"}
                    )
                    st.plotly_chart(styled_fig(fig, f"Top 15 Features"), use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 6 — OPTIMIZATION SIMULATOR
# ═══════════════════════════════════════════════════════════════

with tab_opt:
    st.markdown('<div class="section-title">What-If Factory Reallocation Simulator</div>', unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        sim_product  = st.selectbox("Product", all_products, key="sim_prod")
    with col_b:
        sim_region   = st.selectbox("Region", all_regions,   key="sim_reg")
    with col_c:
        sim_shipmode = st.selectbox("Ship Mode", all_modes,   key="sim_sm")

    current_factory = PRODUCT_FACTORY_MAP.get(sim_product, "Unknown")
    st.info(f"**Current Factory:** {current_factory}")

    alt_factories = [f for f in ALL_FACTORIES if f != current_factory]
    alt_factory   = st.selectbox("Alternative Factory to Simulate", alt_factories, key="alt_fac")

    if st.button("▶ Run Scenario Comparison", type="primary"):
        with st.spinner("Running Monte Carlo comparison…"):
            comparison = compare_scenarios(
                sim_product, sim_region, sim_shipmode,
                current_factory, alt_factory, n_runs=1000
            )

        mc_a = comparison["scenario_a"]
        mc_b = comparison["scenario_b"]

        cols = st.columns(4)
        metrics = [
            ("Current Lead Time",     f"{mc_a['mean']:.2f} days",   f"±{mc_a['std']:.2f}"),
            ("Alternative Lead Time", f"{mc_b['mean']:.2f} days",   f"±{mc_b['std']:.2f}"),
            ("Lead Time Improvement", f"{comparison['improvement_pct']:.1f}%", "vs current"),
            ("Recommended Factory",   comparison["recommendation"],  "optimal choice"),
        ]
        for col, (label, val, note) in zip(cols, metrics):
            with col:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-value">{val}</div>
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-delta">{note}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Distribution comparison
        import plotly.figure_factory as ff
        samples_a = mc_a["samples"]
        samples_b = mc_b["samples"]
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=samples_a, name=f"Current ({current_factory})",
            opacity=0.7, marker_color="#FF6B6B", nbinsx=30,
            histnorm="probability density"
        ))
        fig.add_trace(go.Histogram(
            x=samples_b, name=f"Alternative ({alt_factory})",
            opacity=0.7, marker_color="#64FFDA", nbinsx=30,
            histnorm="probability density"
        ))
        fig.add_vline(x=mc_a["mean"], line_dash="dash", line_color="#FF6B6B",
                      annotation_text=f"Current μ={mc_a['mean']:.2f}")
        fig.add_vline(x=mc_b["mean"], line_dash="dash", line_color="#64FFDA",
                      annotation_text=f"Alt μ={mc_b['mean']:.2f}")
        st.plotly_chart(styled_fig(fig, "📊 Lead Time Distribution Comparison (1000 simulations)"),
                        use_container_width=True)

        # Confidence interval comparison
        ci_data = pd.DataFrame({
            "Scenario":   [f"Current\n({current_factory})", f"Alternative\n({alt_factory})"],
            "Mean":       [mc_a["mean"], mc_b["mean"]],
            "CI_Low":     [mc_a["ci_95_low"], mc_b["ci_95_low"]],
            "CI_High":    [mc_a["ci_95_high"], mc_b["ci_95_high"]],
        })
        fig = go.Figure()
        for _, row in ci_data.iterrows():
            fig.add_trace(go.Scatter(
                x=[row["Scenario"], row["Scenario"]],
                y=[row["CI_Low"], row["CI_High"]],
                mode="lines", line=dict(color="#8892B0", width=3),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=[row["Scenario"]], y=[row["Mean"]],
                mode="markers", marker=dict(size=14, color="#7EB8F7"),
                name=row["Scenario"]
            ))
        st.plotly_chart(styled_fig(fig, "95% Confidence Interval Comparison"),
                        use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 7 — RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════

with tab_rec:
    st.markdown('<div class="section-title">🏆 Top Factory Reallocation Recommendations</div>', unsafe_allow_html=True)

    if rec.empty:
        st.warning("No recommendations available. Ensure models are trained.")
    else:
        # Summary KPIs
        k1, k2, k3, k4 = st.columns(4)
        avg_lt_red = rec["LT_Reduction_Pct"].mean()
        avg_pr_imp = rec["Profit_Improve_Pct"].mean()
        avg_risk   = rec["Risk_Reduction_Pct"].mean()
        avg_score  = rec["Optimization_Score"].mean()
        kpi_data = [
            (k1, "Avg Lead Time Reduction", f"{avg_lt_red:.1f}%", ""),
            (k2, "Avg Profit Improvement",  f"{avg_pr_imp:.1f}%", ""),
            (k3, "Avg Risk Reduction",       f"{avg_risk:.1f}%",  ""),
            (k4, "Avg Optimization Score",   f"{avg_score:.1f}/100",""),
        ]
        for col, label, val, note in kpi_data:
            with col:
                st.markdown(f"""<div class="kpi-card">
                    <div class="kpi-value">{val}</div>
                    <div class="kpi-label">{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Top 10 recommendations
        col1, col2 = st.columns([1.8, 1.2])
        with col1:
            st.markdown("#### Top 10 Recommendations by Optimization Score")
            display = rec.head(10)[[
                "Product","Region","Ship_Mode","Current_Factory","Recommended_Factory",
                "LT_Reduction_Pct","Profit_Improve_Pct","Risk_Reduction_Pct",
                "Confidence_Score","Optimization_Score"
            ]].reset_index(drop=True)
            display.index = display.index + 1
            st.dataframe(
                display.style
                    .background_gradient(subset=["Optimization_Score"], cmap="Blues")
                    .format({
                        "LT_Reduction_Pct": "{:.1f}%",
                        "Profit_Improve_Pct": "{:.1f}%",
                        "Risk_Reduction_Pct": "{:.1f}%",
                        "Confidence_Score": "{:.1f}%",
                        "Optimization_Score": "{:.1f}",
                    }),
                use_container_width=True
            )

        with col2:
            fig = px.scatter(rec.head(30),
                             x="LT_Reduction_Pct", y="Profit_Improve_Pct",
                             size="Optimization_Score", color="Risk_Reduction_Pct",
                             hover_data=["Product","Region","Recommended_Factory"],
                             color_continuous_scale="RdYlGn",
                             labels={"LT_Reduction_Pct":"Lead Time Reduction %",
                                     "Profit_Improve_Pct":"Profit Improvement %"})
            st.plotly_chart(styled_fig(fig, "Optimization Space"), use_container_width=True)

        # Score breakdown chart
        top5 = rec.head(5).copy()
        top5["Label"] = top5["Product"].str[:18] + " → " + top5["Recommended_Factory"].str[:10]
        fig = go.Figure()
        for component, color in [
            ("LT_Reduction_Pct", "#7EB8F7"),
            ("Profit_Improve_Pct", "#64FFDA"),
            ("Risk_Reduction_Pct", "#FFB347"),
            ("Confidence_Score", "#C9B1FF"),
        ]:
            label = component.replace("_Pct","").replace("_"," ")
            fig.add_trace(go.Bar(
                name=label, x=top5["Label"], y=top5[component],
                marker_color=color
            ))
        fig.update_layout(barmode="group")
        st.plotly_chart(styled_fig(fig, "Top 5 Recommendations — Score Components"),
                        use_container_width=True)

        # Download
        csv = rec.to_csv(index=False)
        st.download_button(
            "⬇️ Download All Recommendations (CSV)",
            csv, "recommendations.csv", "text/csv"
        )


# ═══════════════════════════════════════════════════════════════
# TAB 8 — MONTE CARLO
# ═══════════════════════════════════════════════════════════════

with tab_mc:
    st.markdown('<div class="section-title">Monte Carlo Simulation — Uncertainty Analysis</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1: mc_product  = st.selectbox("Product",   all_products, key="mc_prod")
    with col2: mc_region   = st.selectbox("Region",    all_regions,  key="mc_reg")
    with col3: mc_factory  = st.selectbox("Factory",   ALL_FACTORIES, key="mc_fac")

    mc_shipmode = st.selectbox("Ship Mode", all_modes, key="mc_sm")
    mc_runs     = st.slider("Number of Simulations", 100, 5000, 1000, 100)

    if st.button("▶ Run Monte Carlo Simulation", type="primary"):
        with st.spinner(f"Running {mc_runs} simulations…"):
            mc = monte_carlo_simulation(
                mc_product, mc_region, mc_shipmode, mc_factory, mc_runs
            )

        # KPIs
        k1, k2, k3, k4, k5 = st.columns(5)
        for col, (label, val) in zip([k1,k2,k3,k4,k5], [
            ("Expected Lead Time", f"{mc['mean']:.2f} days"),
            ("Best Case (P5)",     f"{mc['p5']:.2f} days"),
            ("Worst Case (P95)",   f"{mc['p95']:.2f} days"),
            ("Std Deviation",      f"±{mc['std']:.2f} days"),
            ("95% CI",             f"[{mc['ci_95_low']:.1f}, {mc['ci_95_high']:.1f}]"),
        ]):
            with col:
                st.markdown(f"""<div class="kpi-card">
                    <div class="kpi-value">{val}</div>
                    <div class="kpi-label">{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Distribution plot
        samples = mc["samples"]
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=samples, name="Simulated Lead Times",
            nbinsx=50, marker_color="#7EB8F7", opacity=0.8,
            histnorm="probability density"
        ))
        # Add percentile lines
        for pct, val, color in [
            ("P5",  mc["p5"],  "#FFB347"),
            ("Mean",mc["mean"],"#64FFDA"),
            ("P95", mc["p95"], "#FF6B6B"),
        ]:
            fig.add_vline(x=val, line_dash="dash", line_color=color,
                          annotation_text=f"{pct}={val:.2f}",
                          annotation_position="top")
        st.plotly_chart(styled_fig(fig, f"Lead Time Distribution — {mc_runs} Monte Carlo Runs"),
                        use_container_width=True)

        # Convergence plot
        running_means = np.cumsum(samples) / np.arange(1, len(samples) + 1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(running_means) + 1)),
            y=running_means, mode="lines",
            line=dict(color="#7EB8F7", width=2),
            name="Running Mean"
        ))
        fig.add_hline(y=mc["mean"], line_dash="dot", line_color="#64FFDA",
                      annotation_text=f"Converged Mean={mc['mean']:.3f}")
        st.plotly_chart(styled_fig(fig, "📉 Simulation Convergence"), use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 9 — SHAP EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════

with tab_shap:
    st.markdown('<div class="section-title">🧠 SHAP Explainability — Model Transparency</div>', unsafe_allow_html=True)

    shap_choice = st.radio("Select Target", ["Lead Time", "Gross Profit"], horizontal=True)
    shap_key    = "lt" if shap_choice == "Lead Time" else "pr"
    shap_data   = get_shap(shap_key)

    if shap_data is None:
        st.warning("SHAP values not found. Run the training script with SHAP enabled.")
    else:
        shap_df = shap_data["shap_df"]

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                shap_df.head(15), x="Mean_Abs_SHAP", y="Feature",
                orientation="h", color="Importance_Pct",
                color_continuous_scale="Blues",
                labels={"Mean_Abs_SHAP":"Mean |SHAP Value|","Importance_Pct":"Importance %"}
            )
            st.plotly_chart(styled_fig(fig, f"Global SHAP Feature Importance — {shap_choice}"),
                            use_container_width=True)

        with col2:
            fig = px.pie(
                shap_df.head(8), names="Feature", values="Importance_Pct",
                color_discrete_sequence=PALETTE, hole=0.4
            )
            fig.update_traces(textinfo="label+percent")
            st.plotly_chart(styled_fig(fig, "Feature Contribution Share"),
                            use_container_width=True)

        # SHAP beeswarm-style (using scatter approximation)
        shap_values = shap_data["shap_values"]
        X_sample    = shap_data["X_sample"]
        feat_names  = shap_data["feature_names"]
        n_plot = min(8, len(feat_names))
        top_feats = shap_df["Feature"].head(n_plot).tolist()
        top_idx   = [feat_names.index(f) for f in top_feats]

        rows = []
        for i, (fi, fname) in enumerate(zip(top_idx, top_feats)):
            for sv, fv in zip(shap_values[:, fi], X_sample[:, fi]):
                rows.append({"Feature": fname, "SHAP Value": sv, "Feature Value": fv})
        beeswarm_df = pd.DataFrame(rows)

        beeswarm_df = beeswarm_df.dropna()

fig = px.strip(
    beeswarm_df,
    x="SHAP Value",
    y="Feature",
    color="SHAP Value",
    color_continuous_scale="RdBu_r",
    stripmode="overlay",
)

st.plotly_chart(fig, use_container_width=True)
        st.markdown("### 💬 SHAP Business Interpretations")
        for _, row in shap_df.head(6).iterrows():
            feat = row["Feature"].replace("_"," ")
            pct  = row["Importance_Pct"]
            direction = "positively" if shap_values[:, feat_names.index(row["Feature"])].mean() > 0 else "negatively"
            st.markdown(f"""
            <div class="insight-card">
                <strong>{feat}</strong> drives {pct:.1f}% of {shap_choice} predictions
                and on average {direction} influences the outcome.
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 10 — FACTORY MAP
# ═══════════════════════════════════════════════════════════════

with tab_map:
    st.markdown('<div class="section-title">🗺️ Geographic Intelligence — Factory Map</div>', unsafe_allow_html=True)

    # Factory markers
    factory_df = pd.DataFrame([
        {
            "Factory": name,
            "Latitude": lat,
            "Longitude": lon,
            "Units": df.groupby("Factory")["Units"].sum().get(name, 0),
            "Revenue": df.groupby("Factory")["Sales"].sum().get(name, 0),
        }
        for name, (lat, lon) in FACTORY_COORDS.items()
    ])

    fig = px.scatter_mapbox(
        factory_df,
        lat="Latitude", lon="Longitude",
        size="Units", color="Revenue",
        hover_name="Factory",
        hover_data={"Units":True, "Revenue":":.0f"},
        color_continuous_scale="Blues",
        size_max=40, zoom=3, height=500,
        mapbox_style="carto-darkmatter",
        labels={"Revenue":"Revenue ($)"}
    )

    # Add region markers
    region_df = pd.DataFrame([
        {"Region": r, "Latitude": lat, "Longitude": lon}
        for r, (lat, lon) in REGION_COORDS.items()
    ])
    fig.add_trace(go.Scattermapbox(
        lat=region_df["Latitude"], lon=region_df["Longitude"],
        mode="markers+text",
        marker=dict(size=14, color="#64FFDA", symbol="circle"),
        text=region_df["Region"], textposition="top right",
        name="Sales Regions",
        textfont=dict(color="#64FFDA", size=11)
    ))

    fig.update_layout(
        paper_bgcolor="#1a1f2e",
        font=dict(color="#CDD6F4"),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(bgcolor="#1a1f2e", bordercolor="#3d4166")
    )
    st.plotly_chart(fig, use_container_width=True)

    # Distance table
    st.markdown("### 📏 Factory → Region Distance Summary")
    dist_df2 = compute_all_distances()
    factory_metrics2 = dff.groupby("Factory")["Units"].sum().rename("Total_Units")
    dist_df2 = dist_df2.merge(factory_metrics2.reset_index(), on="Factory", how="left")

    fig2 = px.bar(
        dist_df2, x="Factory", y="Distance_km", color="Region",
        barmode="group", color_discrete_sequence=PALETTE,
        labels={"Distance_km":"Distance (km)"}
    )
    st.plotly_chart(styled_fig(fig2, "Factory-to-Region Shipping Distances"), use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 11 — RISK ANALYTICS
# ═══════════════════════════════════════════════════════════════

with tab_risk:
    st.markdown('<div class="section-title">⚠️ Risk Analytics & Alerts</div>', unsafe_allow_html=True)

    # High-risk routes
    route_risk = dff.groupby(["Factory","Region","Product Name"]).agg(
        Avg_Risk=("Route_Risk","mean"),
        Avg_LT=("Lead_Time","mean"),
        Total_Units=("Units","sum"),
        Avg_Margin=("Profit_Margin","mean"),
    ).reset_index().sort_values("Avg_Risk", ascending=False)

    high_risk = route_risk[route_risk["Avg_Risk"] > route_risk["Avg_Risk"].quantile(0.75)]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🔴 High-Risk Routes (Top Quartile)")
        st.dataframe(
            high_risk.head(15).style
                .background_gradient(subset=["Avg_Risk"], cmap="Reds")
                .format({"Avg_Risk":"{:.4f}","Avg_LT":"{:.2f}","Avg_Margin":"{:.2%}"}),
            use_container_width=True
        )

    with col2:
        fig = px.scatter(route_risk.head(50),
                         x="Avg_LT", y="Avg_Risk",
                         size="Total_Units", color="Factory",
                         hover_data=["Region","Product Name","Avg_Margin"],
                         color_discrete_sequence=PALETTE, size_max=30)
        # Risk threshold line
        threshold = route_risk["Avg_Risk"].quantile(0.75)
        fig.add_hline(y=threshold, line_dash="dash", line_color="#FF6B6B",
                      annotation_text="High Risk Threshold")
        st.plotly_chart(styled_fig(fig, "Risk vs Lead Time (bubble = volume)"), use_container_width=True)

    # Factory dependency risk
    st.markdown("#### 🏭 Factory Concentration Risk")
    factory_conc = dff.groupby("Factory")["Units"].sum()
    factory_conc_pct = (factory_conc / factory_conc.sum() * 100).round(2)

    fig = px.pie(
        factory_conc_pct.reset_index(),
        names="Factory", values="Units",
        color_discrete_sequence=["#FF6B6B","#FFB347","#64FFDA","#7EB8F7","#C9B1FF"],
        hole=0.5
    )
    fig.update_traces(textinfo="label+percent")
    st.plotly_chart(styled_fig(fig, "Unit Concentration by Factory (Dependency Risk)"),
                    use_container_width=True)

    # Margin erosion risk
    st.markdown("#### 💸 Margin Erosion Risk by Route")
    margin_risk = dff.groupby(["Factory","Region"]).agg(
        Avg_Margin=("Profit_Margin","mean"),
        Avg_Distance=("Shipping_Distance_km","mean"),
    ).reset_index()
    margin_risk["Risk_Level"] = pd.cut(
        margin_risk["Avg_Margin"],
        bins=[0, 0.3, 0.5, 1.0],
        labels=["🔴 High Risk","🟡 Medium","🟢 Low Risk"]
    ).astype(str)

    fig = px.scatter(margin_risk,
                     x="Avg_Distance", y="Avg_Margin",
                     color="Risk_Level",
                     color_discrete_map={"🔴 High Risk":"#FF6B6B","🟡 Medium":"#FFB347","🟢 Low Risk":"#64FFDA"},
                     hover_data=["Factory","Region"], size_max=20,
                     labels={"Avg_Margin":"Avg Profit Margin","Avg_Distance":"Avg Distance (km)"})
    fig.add_hline(y=0.3, line_dash="dash", line_color="#FF6B6B", annotation_text="Low Margin Threshold")
    st.plotly_chart(styled_fig(fig, "Margin Erosion vs Shipping Distance"), use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 12 — EXECUTIVE REPORT
# ═══════════════════════════════════════════════════════════════

with tab_report:
    st.markdown('<div class="section-title">📄 Executive Report</div>', unsafe_allow_html=True)

    total_revenue = dff["Sales"].sum()
    total_profit  = dff["Gross Profit"].sum()
    avg_margin    = dff["Profit_Margin"].mean()
    avg_lt        = dff["Lead_Time"].mean()
    top_rec       = rec.iloc[0] if not rec.empty else None

    report_md = f"""
## SupplyChainAI — Executive Summary
### Nassau Candy Distributor | Intelligent Factory Reallocation Analysis

---

### 1. Business Context

Nassau Candy currently operates with **static factory-to-product assignments** established
through legacy processes. This analysis demonstrates that reallocation of specific product
families to alternative manufacturing facilities can yield measurable improvements in
shipping lead time, logistics risk, and profit margins.

---

### 2. Current State Performance

| Metric | Value |
|--------|-------|
| Total Revenue (filtered) | ${total_revenue:,.0f} |
| Total Gross Profit | ${total_profit:,.0f} |
| Average Profit Margin | {avg_margin*100:.1f}% |
| Average Lead Time | {avg_lt:.1f} days |
| Total Orders Analyzed | {len(dff):,} |
| Active Products | {dff["Product Name"].nunique()} |
| Sales Regions | {dff["Region"].nunique()} |

---

### 3. Critical Findings

**Factory Imbalance:** Lot's O' Nuts handles **55.7% of all units** — a critical single-point
dependency. Any disruption to this facility directly impacts the majority of SKUs.

**Geographic Inefficiency:** Products shipped from Lot's O' Nuts (Arizona) to the Pacific
region travel only **~1,108 km** — the shortest route. However, the same factory shipping
to the Atlantic region covers **3,445 km** — the longest route, adding ~2.3 days in lead time.

**Sugar Shack Underutilisation:** Currently processing only **0.4% of total units** despite
being geographically central, Sugar Shack represents a significant untapped capacity resource.

---

### 4. Top Optimization Opportunities

| Rank | Product | Recommended Move | Lead Time Reduction | Optimization Score |
|------|---------|------------------|--------------------|--------------------|
{''.join([f"| {i+1} | {r['Product']} | {r['Current_Factory']} → {r['Recommended_Factory']} | {r['LT_Reduction_Pct']:.1f}% | {r['Optimization_Score']:.1f}/100 |" + chr(10) for i, (_, r) in enumerate(rec.head(5).iterrows())]) if not rec.empty else "| – | No recommendations available | – | – | – |"}

---

### 5. Model Performance

| Target | Best Model | MAE | R² Score |
|--------|-----------|-----|----------|
| Lead Time | {art["lead_time"]["model_name"] if art else "N/A"} | {art["lead_time"]["results"].get(art["lead_time"]["model_name"],{}).get("MAE_test","N/A") if art else "N/A"} | {art["lead_time"]["results"].get(art["lead_time"]["model_name"],{}).get("R2_test","N/A") if art else "N/A"} |
| Gross Profit | {art["profit"]["model_name"] if art else "N/A"} | {art["profit"]["results"].get(art["profit"]["model_name"],{}).get("MAE_test","N/A") if art else "N/A"} | {art["profit"]["results"].get(art["profit"]["model_name"],{}).get("R2_test","N/A") if art else "N/A"} |

---

### 6. Strategic Recommendations

1. **Reallocate Laffy Taffy and SweeTARTS** from Sugar Shack to Secret Factory or
   Wicked Choccy's for Pacific and Atlantic shipments — projected lead time reduction of 18–29%.

2. **Redistribute Wonka Bar variants** currently at Lot's O' Nuts to reduce Atlantic/Interior
   route risk by shifting partial volume to The Other Factory.

3. **Invest in Sugar Shack capacity expansion** — its central US location makes it ideal for
   serving Interior and Gulf regions with minimal lead time.

4. **Implement dynamic ship mode optimisation** — switching 20% of Standard Class volume
   to Second Class for high-margin products would reduce average lead time by ~2 days.

---

### 7. Implementation Roadmap

**Phase 1 (0–3 months):** Pilot reallocation of SweeTARTS and Laffy Taffy to Secret Factory
for Interior region orders. Measure actual vs predicted lead time improvement.

**Phase 2 (3–6 months):** Scale Sugar Shack capacity for Gulf and Interior sugar product lines.
Implement ship mode recommendation engine at order entry.

**Phase 3 (6–12 months):** Full deployment of SupplyChainAI recommendation engine into
ERP/WMS workflow. Continuous model retraining on live order data.

---

*Report generated by SupplyChainAI v1.0 — Powered by Gradient Boosting, Monte Carlo Simulation & SHAP Explainability*
"""

    st.markdown(report_md)

    # Download
    st.download_button(
        "⬇️ Download Executive Report (Markdown)",
        report_md,
        "nassau_candy_executive_report.md",
        "text/markdown"
    )
