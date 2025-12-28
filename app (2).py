import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium
# CONFIG
st.set_page_config(page_title="Prescriptive Map", layout="wide")
st.title("🚚 Prescriptive Map – Last-mile Delivery 🚚")

st.markdown(
    """
    <div style="
        display:inline-block;
        color:#000;
        background:rgba(255,255,255,0.92);
        padding:6px 10px;
        border-radius:10px;
        border:1px solid rgba(0,0,0,0.12);
        font-size:14px;
        font-weight:600;
        margin-top:-6px;
        margin-bottom:12px;
    ">
        From prediction → decision
    </div>
    """,
    unsafe_allow_html=True
)

@st.cache_data
def load_data():
    return pd.read_parquet("eta_region_geo.parquet")

eta_region_geo = load_data()

# CONSTANTS
CAPACITY_PER_COURIER = 30
SLA_ETA_MINUTES = 60
K = 3

# PRESCRIPTIVE LOGIC
def build_prescriptive_decisions(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()

    df["rec_couriers"] = np.ceil(
        df["demand_mean"] / CAPACITY_PER_COURIER
    ).astype(int)

    d_norm = df.groupby("city")["demand_mean"].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
    )
    e_norm = df.groupby("city")["expected_eta"].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
    )

    df["priority_score"] = 0.6 * d_norm + 0.4 * e_norm

    q_hi = df.groupby("city")["priority_score"].transform("quantile", 0.67)
    q_lo = df.groupby("city")["priority_score"].transform("quantile", 0.33)

    df["action"] = np.where(
        df["priority_score"] >= q_hi, "PRIORITIZE",
        np.where(df["priority_score"] <= q_lo, "DE-PRIORITIZE", "MAINTAIN")
    )

    df["sla_risk"] = (df["expected_eta_p90"] > SLA_ETA_MINUTES).astype(int)
    return df

# SIDEBAR
viz_date = st.sidebar.selectbox( "Date", sorted(eta_region_geo["date"].astype(str).unique()) )
# DATA – TOP K FIXED
df_day = (
    eta_region_geo.query("date == @viz_date")
    .sort_values(["city", "demand_mean"], ascending=[True, False])
    .groupby("city")
    .head(K)
    .copy()
)

df_day = build_prescriptive_decisions(df_day)

# BASE MAP
m = folium.Map(
    location=[df_day["lat"].mean(), df_day["lng"].mean()],
    zoom_start=5,
    tiles="cartodbpositron"
)

# TITLE BOX TRONG MAP
n_city = df_day["city"].nunique()
k_real = int(df_day.groupby("city")["region_id"].nunique().max())

m.get_root().html.add_child(folium.Element(f"""
<div style="position: fixed; top: 10px; left: 10px; z-index: 9999;
            background: rgba(255,255,255,0.92); padding: 10px 12px;
            border-radius: 10px; border: 1px solid rgba(0,0,0,0.18);
            font-size: 13px;">
  <b>Prescriptive Maps (All Cities)</b><br/>
  Date: <b>{viz_date}</b><br/>
  Coverage: <b>{n_city} cities × top {k_real} regions</b><br/>
  CAPACITY_PER_COURIER: <b>{CAPACITY_PER_COURIER}</b> orders/courier/day<br/>
  SLA threshold (ETA p90): <b>{SLA_ETA_MINUTES} min</b>
</div>
"""))

# A) PRIORITY
priority_layer = folium.FeatureGroup(
    name="A) Priority (action + score)", show=True
)
mc_a = MarkerCluster().add_to(priority_layer)

color_action = {
    "PRIORITIZE": "red",
    "MAINTAIN": "blue",
    "DE-PRIORITIZE": "green"
}

for _, r in df_day.iterrows():
    folium.CircleMarker(
        location=[r.lat, r.lng],
        radius=6 + 14 * r.priority_score,
        color=color_action[r.action],
        fill=True, fill_opacity=0.65,
        tooltip=f"{r.city} | R{r.region_id} | {r.action} | score={r.priority_score:.2f}"
    ).add_to(mc_a)

priority_layer.add_to(m)

# B) CAPACITY
capacity_layer = folium.FeatureGroup(
    name="B) Capacity (recommended couriers)", show=False
)
mc_b = MarkerCluster().add_to(capacity_layer)

max_c = max(1, int(df_day["rec_couriers"].max()))
for _, r in df_day.iterrows():
    rad = 5 + 20 * (r.rec_couriers / max_c)
    folium.CircleMarker(
        location=[r.lat, r.lng],
        radius=rad,
        color="black",
        fill=True, fill_opacity=0.5,
        tooltip=f"{r.city} | R{r.region_id} | couriers={int(r.rec_couriers)}"
    ).add_to(mc_b)

capacity_layer.add_to(m)

# C) RISK
risk_layer = folium.FeatureGroup(
    name="C) Risk (ETA p90 > SLA?)", show=False
)
mc_c = MarkerCluster().add_to(risk_layer)

for _, r in df_day.iterrows():
    col = "orange" if r.sla_risk == 1 else "cadetblue"
    folium.CircleMarker(
        location=[r.lat, r.lng],
        radius=7,
        color=col,
        fill=True, fill_opacity=0.75,
        tooltip=f"{r.city} | R{r.region_id} | ETA_p90={r.expected_eta_p90:.0f}"
    ).add_to(mc_c)

risk_layer.add_to(m)

# D) HEATMAP
heat_layer = folium.FeatureGroup(
    name="D) Heatmap (priority pressure)", show=False
)
heat_data = [
    [r.lat, r.lng, r.priority_score]
    for _, r in df_day.iterrows()
]
HeatMap(heat_data, radius=22, blur=18, min_opacity=0.25).add_to(heat_layer)
heat_layer.add_to(m)
# E) PERSISTENCE
persist_layer = folium.FeatureGroup(
    name="E) Persistence (count of PRIORITIZE days)", show=False
)

df_all = (
    eta_region_geo
    .sort_values(["date", "city", "demand_mean"], ascending=[True, True, False])
    .groupby(["date", "city"])
    .head(K)
    .copy()
)

df_all = build_prescriptive_decisions(df_all)

persist = (
    df_all.assign(is_prioritize=(df_all["action"] == "PRIORITIZE").astype(int))
    .groupby(["city", "region_id", "lat", "lng"], as_index=False)
    .agg(
        prioritize_days=("is_prioritize", "sum"),
        total_days=("date", "nunique")
    )
)

persist["prioritize_rate"] = (
    persist["prioritize_days"] /
    persist["total_days"].clip(lower=1)
)

max_days = max(1, int(persist["prioritize_days"].max()))
for _, r in persist.iterrows():
    rad = 4 + 18 * (r.prioritize_days / max_days)
    folium.CircleMarker(
        location=[r.lat, r.lng],
        radius=rad,
        color="purple",
        fill=True, fill_opacity=0.55,
        tooltip=f"{r.city} | R{r.region_id} | {int(r.prioritize_days)}/{int(r.total_days)}"
    ).add_to(persist_layer)

persist_layer.add_to(m)

# CONTROLS + RENDER
folium.LayerControl(collapsed=False).add_to(m)
st_folium(m, use_container_width=True, height=650)
