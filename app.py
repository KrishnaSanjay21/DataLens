import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="DataLens — EDA Report Generator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem !important; max-width: 100% !important; }

/* Header */
.app-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.75rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}
.app-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: #f0eff5;
    letter-spacing: -0.025em;
    margin: 0;
}
.app-title span { color: #22d3ee; }
.app-badge {
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    background: rgba(34,211,238,0.1);
    color: #22d3ee;
    border: 1px solid rgba(34,211,238,0.25);
    padding: 3px 10px;
    border-radius: 999px;
}

/* Cards */
.stat-card {
    background: #111118;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1rem 1.25rem;
    height: 100%;
}
.stat-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #666;
    margin-bottom: 4px;
}
.stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: #22d3ee;
    letter-spacing: -0.02em;
    line-height: 1.1;
}
.stat-sub { font-size: 0.72rem; color: #555; margin-top: 3px; }

/* Section headers */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #f0eff5;
    margin: 2rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-title::before {
    content: '';
    display: inline-block;
    width: 3px; height: 16px;
    background: #22d3ee;
    border-radius: 2px;
}

/* Issue badges */
.badge-warn {
    display: inline-block;
    background: rgba(251,191,36,0.1);
    color: #fbbf24;
    border: 1px solid rgba(251,191,36,0.25);
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 500;
    margin: 2px;
}
.badge-ok {
    display: inline-block;
    background: rgba(52,211,153,0.1);
    color: #34d399;
    border: 1px solid rgba(52,211,153,0.25);
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 500;
    margin: 2px;
}
.badge-err {
    display: inline-block;
    background: rgba(239,68,68,0.1);
    color: #f87171;
    border: 1px solid rgba(239,68,68,0.25);
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 500;
    margin: 2px;
}

/* Column type chip */
.col-chip {
    display: inline-block;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.7rem;
    color: #888;
    margin: 2px;
}

/* Insight box */
.insight-box {
    background: rgba(34,211,238,0.05);
    border: 1px solid rgba(34,211,238,0.15);
    border-left: 3px solid #22d3ee;
    border-radius: 0 10px 10px 0;
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    color: #c8c7d4;
    line-height: 1.65;
    margin: 0.5rem 0;
}
.insight-box strong { color: #f0eff5; }

/* Warning box */
.warn-box {
    background: rgba(251,191,36,0.05);
    border: 1px solid rgba(251,191,36,0.15);
    border-left: 3px solid #fbbf24;
    border-radius: 0 10px 10px 0;
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    color: #c8c7d4;
    line-height: 1.65;
    margin: 0.5rem 0;
}

/* Upload area */
.upload-zone {
    background: rgba(34,211,238,0.03);
    border: 1px dashed rgba(34,211,238,0.2);
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
}

/* Buttons */
.stButton > button {
    background: #22d3ee !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #06b6d4 !important;
    transform: translateY(-1px) !important;
}

/* Radio / selectbox */
div[data-testid="stRadio"] label { font-size: 0.875rem !important; }

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #666 !important;
}
.stTabs [aria-selected="true"] { color: #22d3ee !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ── Chart theme ───────────────────────────────────────────────
BG = "#0a0a0f"
SURFACE = "#111118"
GRID = "rgba(255,255,255,0.05)"
TEXT = "#888797"
PALETTE = ["#22d3ee","#fbbf24","#a78bfa","#34d399","#f472b6","#fb923c","#60a5fa","#e879f9"]

def theme(fig, title=""):
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family="DM Sans", color=TEXT, size=11),
        title=dict(text=title, font=dict(family="Syne", color="#f0eff5", size=13)),
        margin=dict(l=16, r=16, t=40 if title else 16, b=16),
        xaxis=dict(gridcolor=GRID, linecolor=GRID, tickfont=dict(size=10)),
        yaxis=dict(gridcolor=GRID, linecolor=GRID, tickfont=dict(size=10)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        colorway=PALETTE,
    )
    return fig

# ── Demo datasets ─────────────────────────────────────────────
@st.cache_data
def load_titanic():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    try:
        df = pd.read_csv(url)
        return df
    except:
        # Fallback: generate synthetic Titanic-like data
        np.random.seed(42)
        n = 891
        df = pd.DataFrame({
            "PassengerId": range(1, n+1),
            "Survived": np.random.choice([0,1], n, p=[0.62, 0.38]),
            "Pclass": np.random.choice([1,2,3], n, p=[0.24,0.21,0.55]),
            "Name": [f"Passenger_{i}" for i in range(n)],
            "Sex": np.random.choice(["male","female"], n, p=[0.65,0.35]),
            "Age": np.where(np.random.rand(n) < 0.2, np.nan,
                           np.random.normal(30, 14, n).clip(1, 80)),
            "SibSp": np.random.choice(range(6), n, p=[0.68,0.23,0.06,0.02,0.005,0.005]),
            "Parch": np.random.choice(range(7), n, p=[0.76,0.13,0.08,0.02,0.005,0.002,0.003]),
            "Fare": np.abs(np.random.lognormal(3.2, 1.0, n)).round(2),
            "Embarked": np.random.choice(["S","C","Q",np.nan], n, p=[0.72,0.19,0.086,0.004]),
            "Cabin": np.where(np.random.rand(n) < 0.77, np.nan, "C123"),
        })
        return df

@st.cache_data
def load_iris():
    from sklearn.datasets import load_iris as _load
    data = _load()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["species"] = pd.Categorical.from_codes(data.target, data.target_names)
    return df

@st.cache_data
def load_nyc():
    np.random.seed(99)
    n = 5000
    boroughs = ["Manhattan","Brooklyn","Queens","Bronx","Staten Island"]
    df = pd.DataFrame({
        "pickup_hour": np.random.choice(range(24), n),
        "pickup_day": np.random.choice(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], n),
        "pickup_borough": np.random.choice(boroughs, n, p=[0.55,0.20,0.15,0.07,0.03]),
        "dropoff_borough": np.random.choice(boroughs, n, p=[0.55,0.20,0.15,0.07,0.03]),
        "passenger_count": np.random.choice(range(1,7), n),
        "trip_distance_miles": np.abs(np.random.lognormal(0.9, 0.8, n)).clip(0.1, 35).round(2),
        "fare_amount": np.abs(np.random.normal(14, 8, n)).clip(2.5, 120).round(2),
        "tip_amount": np.abs(np.random.normal(2.5, 2, n)).clip(0, 30).round(2),
        "payment_type": np.random.choice(["Credit Card","Cash","No Charge"], n, p=[0.67,0.31,0.02]),
        "speed_mph": np.abs(np.random.normal(18, 8, n)).clip(1, 60).round(1),
    })
    return df

# ── Core EDA functions ────────────────────────────────────────
def get_column_types(df):
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=["object","category","bool"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime"]).columns.tolist()
    return numeric, categorical, datetime_cols

def detect_issues(df):
    issues = []
    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(1)
    for col in df.columns:
        if missing[col] > 0:
            severity = "high" if missing_pct[col] > 30 else "medium" if missing_pct[col] > 5 else "low"
            issues.append({
                "column": col, "issue": "Missing values",
                "detail": f"{missing[col]:,} missing ({missing_pct[col]}%)",
                "severity": severity
            })
    # Duplicates
    dupes = df.duplicated().sum()
    if dupes > 0:
        issues.append({
            "column": "All", "issue": "Duplicate rows",
            "detail": f"{dupes:,} duplicate rows ({dupes/len(df)*100:.1f}%)",
            "severity": "high" if dupes/len(df) > 0.05 else "medium"
        })
    # Outliers (IQR method)
    numeric, _, _ = get_column_types(df)
    for col in numeric:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < Q1 - 3*IQR) | (df[col] > Q3 + 3*IQR)).sum()
        if outliers > 0:
            issues.append({
                "column": col, "issue": "Outliers (3×IQR)",
                "detail": f"{outliers:,} extreme outliers ({outliers/len(df)*100:.1f}%)",
                "severity": "medium" if outliers/len(df) > 0.01 else "low"
            })
    # High cardinality
    _, categorical, _ = get_column_types(df)
    for col in categorical:
        n_unique = df[col].nunique()
        if n_unique > 50:
            issues.append({
                "column": col, "issue": "High cardinality",
                "detail": f"{n_unique:,} unique values",
                "severity": "low"
            })
    # Constant / near-constant columns
    for col in df.columns:
        if df[col].nunique() <= 1:
            issues.append({
                "column": col, "issue": "Constant column",
                "detail": "Only 1 unique value — likely useless for modeling",
                "severity": "high"
            })
    # Skewness
    for col in numeric:
        skew = df[col].skew()
        if abs(skew) > 2:
            issues.append({
                "column": col, "issue": "High skewness",
                "detail": f"Skewness = {skew:.2f} — consider log transform",
                "severity": "low"
            })
    return issues

def generate_insights(df):
    numeric, categorical, _ = get_column_types(df)
    insights = []

    # Dataset size
    insights.append(f"Dataset contains <strong>{len(df):,} rows</strong> and <strong>{len(df.columns)} columns</strong> — {len(numeric)} numeric, {len(categorical)} categorical.")

    # Missing data summary
    total_missing = df.isnull().sum().sum()
    total_cells = df.size
    if total_missing > 0:
        insights.append(f"<strong>{total_missing:,} missing values</strong> detected across the dataset ({total_missing/total_cells*100:.1f}% of all cells). Columns with >30% missing may need to be dropped.")
    else:
        insights.append("✅ <strong>No missing values</strong> — this is a clean dataset!")

    # Numeric insights
    for col in numeric[:3]:
        skew = df[col].skew()
        if abs(skew) > 1:
            direction = "right (positive)" if skew > 0 else "left (negative)"
            insights.append(f"<strong>{col}</strong> is skewed {direction} (skewness={skew:.2f}). Consider log or sqrt transformation for ML models.")

    # Correlation insight
    if len(numeric) >= 2:
        corr_matrix = df[numeric].corr().abs()
        corr_vals = corr_matrix.values.copy()
        np.fill_diagonal(corr_vals, 0)
        corr_matrix = pd.DataFrame(corr_vals, index=corr_matrix.index, columns=corr_matrix.columns)
        max_corr = corr_matrix.max().max()
        if max_corr > 0.8:
            idx = corr_matrix.stack().idxmax()
            insights.append(f"Strong correlation detected between <strong>{idx[0]}</strong> and <strong>{idx[1]}</strong> (r={max_corr:.2f}). Consider removing one to avoid multicollinearity.")

    # Class imbalance check
    for col in categorical[:2]:
        vc = df[col].value_counts(normalize=True)
        if vc.iloc[0] > 0.85:
            insights.append(f"<strong>{col}</strong> is heavily imbalanced — top value '{vc.index[0]}' accounts for {vc.iloc[0]*100:.0f}% of records.")

    return insights

# ── App state ─────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = ""

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <span style="font-size:1.75rem">🔬</span>
  <div>
    <div class="app-title">Data<span>Lens</span></div>
  </div>
  <span class="app-badge">EDA Report Generator</span>
  <span style="margin-left:auto; font-size:0.75rem; color:#444;">
    Instant profiling · Zero setup · No API key needed
  </span>
</div>
""", unsafe_allow_html=True)

# ── Data source selection ─────────────────────────────────────
col_src, col_info = st.columns([1.2, 1])

with col_src:
    st.markdown("#### 📂 Load your data")
    source = st.radio(
        "Source",
        ["🗂️ Upload CSV", "🚢 Titanic (demo)", "🌸 Iris (demo)", "🚕 NYC Taxi (demo)"],
        horizontal=True,
        label_visibility="collapsed"
    )

    if source == "🗂️ Upload CSV":
        uploaded = st.file_uploader(
            "Drop a CSV file",
            type=["csv"],
            label_visibility="collapsed"
        )
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                st.session_state.df = df
                st.session_state.dataset_name = uploaded.name
                st.success(f"✅ Loaded **{uploaded.name}** — {len(df):,} rows × {len(df.columns)} columns")
            except Exception as e:
                st.error(f"Could not read file: {e}")
    else:
        demo_map = {
            "🚢 Titanic (demo)": ("Titanic — Passenger Survival", load_titanic),
            "🌸 Iris (demo)": ("Iris — Flower Measurements", load_iris),
            "🚕 NYC Taxi (demo)": ("NYC Taxi Trips", load_nyc),
        }
        name, loader = demo_map[source]
        if st.button(f"Load {source.split(' ')[1]} dataset"):
            with st.spinner("Loading..."):
                st.session_state.df = loader()
                st.session_state.dataset_name = name
            st.success(f"✅ Loaded **{name}**")

# ── Main report ───────────────────────────────────────────────
df = st.session_state.df

if df is None:
    st.markdown("""
    <div style="text-align:center; padding:4rem 0; color:#444;">
        <div style="font-size:3rem; margin-bottom:1rem">🔬</div>
        <div style="font-family:'Syne',sans-serif; font-size:1.1rem; color:#666; margin-bottom:0.5rem">
            No data loaded yet
        </div>
        <div style="font-size:0.85rem">
            Upload a CSV or choose a demo dataset above to generate your report
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

numeric, categorical, datetime_cols = get_column_types(df)
issues = detect_issues(df)
insights = generate_insights(df)

# ── Overview metrics ──────────────────────────────────────────
st.markdown(
    f'<div class="section-title">📊 Dataset Overview — {st.session_state.dataset_name}</div>',
    unsafe_allow_html=True
)

total_missing = df.isnull().sum().sum()
missing_pct = round(total_missing / df.size * 100, 1)
dupes = df.duplicated().sum()
mem_mb = round(df.memory_usage(deep=True).sum() / 1024**2, 2)

m1, m2, m3, m4, m5, m6 = st.columns(6)
metrics = [
    (m1, "Rows", f"{len(df):,}", "records"),
    (m2, "Columns", str(len(df.columns)), f"{len(numeric)} num · {len(categorical)} cat"),
    (m3, "Missing", f"{missing_pct}%", f"{total_missing:,} cells"),
    (m4, "Duplicates", f"{dupes:,}", f"{dupes/len(df)*100:.1f}% of rows"),
    (m5, "Issues Found", str(len(issues)), "detected"),
    (m6, "Memory", f"{mem_mb} MB", "in memory"),
]
for col, label, value, sub in metrics:
    with col:
        color = "#f87171" if (label == "Missing" and missing_pct > 10) or \
                             (label == "Duplicates" and dupes > 0) else \
                "#fbbf24" if label == "Issues Found" and len(issues) > 0 else "#22d3ee"
        st.markdown(
            f'<div class="stat-card">'
            f'<div class="stat-label">{label}</div>'
            f'<div class="stat-value" style="color:{color}">{value}</div>'
            f'<div class="stat-sub">{sub}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

# ── Auto Insights ─────────────────────────────────────────────
st.markdown('<div class="section-title">💡 Auto Insights</div>', unsafe_allow_html=True)
for ins in insights:
    st.markdown(f'<div class="insight-box">{ins}</div>', unsafe_allow_html=True)

# ── Issues ────────────────────────────────────────────────────
if issues:
    st.markdown('<div class="section-title">⚠️ Data Quality Issues</div>', unsafe_allow_html=True)
    issues_df = pd.DataFrame(issues)
    high = issues_df[issues_df.severity == "high"]
    med = issues_df[issues_df.severity == "medium"]
    low = issues_df[issues_df.severity == "low"]

    for severity, grp, badge_cls, emoji in [
        ("High", high, "badge-err", "🔴"),
        ("Medium", med, "badge-warn", "🟡"),
        ("Low", low, "badge-ok", "🟢"),
    ]:
        if not grp.empty:
            st.markdown(f"**{emoji} {severity} severity**")
            for _, row in grp.iterrows():
                st.markdown(
                    f'<span class="{badge_cls}">{row["column"]}</span> '
                    f'<span style="font-size:0.82rem;color:#888">{row["issue"]} — {row["detail"]}</span>',
                    unsafe_allow_html=True
                )
            st.markdown("")
else:
    st.markdown(
        '<div class="insight-box">✅ <strong>No data quality issues found!</strong> This dataset looks clean.</div>',
        unsafe_allow_html=True
    )

# ── Tabs ──────────────────────────────────────────────────────
tabs = st.tabs(["📈 Distributions", "🔗 Correlations", "🗂️ Categorical", "🔢 Statistics", "🧬 Raw Data"])

# ── TAB 1: Distributions ──────────────────────────────────────
with tabs[0]:
    if not numeric:
        st.info("No numeric columns found.")
    else:
        cols_per_row = 3
        for i in range(0, len(numeric), cols_per_row):
            row_cols = st.columns(cols_per_row)
            for j, col in enumerate(numeric[i:i+cols_per_row]):
                with row_cols[j]:
                    series = df[col].dropna()
                    skew = series.skew()
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=series, nbinsx=30,
                        marker_color="#22d3ee",
                        marker_line_color="rgba(0,0,0,0)",
                        opacity=0.85,
                        name=col
                    ))
                    # Add mean/median lines
                    fig.add_vline(x=series.mean(), line_dash="dash",
                                  line_color="#fbbf24", line_width=1.5,
                                  annotation_text="mean", annotation_font_size=9)
                    fig.add_vline(x=series.median(), line_dash="dot",
                                  line_color="#a78bfa", line_width=1.5,
                                  annotation_text="median", annotation_font_size=9)
                    theme(fig, f"{col} (skew={skew:.2f})")
                    st.plotly_chart(fig, use_container_width=True,
                                    config={"displayModeBar": False})

        # Box plots
        st.markdown('<div class="section-title">📦 Box Plots — Outlier View</div>', unsafe_allow_html=True)
        box_cols = numeric[:8]
        fig_box = go.Figure()
        for i, col in enumerate(box_cols):
            fig_box.add_trace(go.Box(
                y=df[col].dropna(),
                name=col,
                marker_color=PALETTE[i % len(PALETTE)],
                boxmean=True,
                line_width=1.5,
            ))
        theme(fig_box, "Box Plots — All Numeric Columns")
        fig_box.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True,
                        config={"displayModeBar": False})

# ── TAB 2: Correlations ───────────────────────────────────────
with tabs[1]:
    if len(numeric) < 2:
        st.info("Need at least 2 numeric columns for correlation analysis.")
    else:
        corr = df[numeric].corr()

        # Heatmap
        fig_heat = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale=[
                [0.0, "#1e0a3c"],
                [0.25, "#312e81"],
                [0.5, "#111118"],
                [0.75, "#164e63"],
                [1.0, "#22d3ee"],
            ],
            zmid=0,
            text=corr.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 9},
            showscale=True,
        ))
        theme(fig_heat, "Correlation Matrix")
        fig_heat.update_layout(height=max(350, len(numeric) * 45))
        st.plotly_chart(fig_heat, use_container_width=True,
                        config={"displayModeBar": False})

        # Top correlations table
        st.markdown('<div class="section-title">🔝 Strongest Correlations</div>', unsafe_allow_html=True)
        corr_pairs = (
            corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            .stack()
            .reset_index()
        )
        corr_pairs.columns = ["Feature A", "Feature B", "Correlation"]
        corr_pairs["Abs"] = corr_pairs["Correlation"].abs()
        corr_pairs = corr_pairs.sort_values("Abs", ascending=False).head(10).drop("Abs", axis=1)
        corr_pairs["Correlation"] = corr_pairs["Correlation"].round(3)
        corr_pairs["Strength"] = corr_pairs["Correlation"].abs().apply(
            lambda x: "🔴 Strong" if x > 0.7 else "🟡 Moderate" if x > 0.4 else "🟢 Weak"
        )
        st.dataframe(corr_pairs, use_container_width=True, hide_index=True)

        # Scatter matrix for top features
        top_features = numeric[:min(5, len(numeric))]
        st.markdown('<div class="section-title">🔵 Scatter Matrix</div>', unsafe_allow_html=True)
        color_col = categorical[0] if categorical else None
        fig_scatter = px.scatter_matrix(
            df[top_features + ([color_col] if color_col else [])].dropna(),
            dimensions=top_features,
            color=color_col,
            color_discrete_sequence=PALETTE,
            opacity=0.5,
        )
        fig_scatter.update_traces(marker=dict(size=3))
        theme(fig_scatter, "Scatter Matrix — Top Numeric Features")
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True,
                        config={"displayModeBar": False})

# ── TAB 3: Categorical ────────────────────────────────────────
with tabs[2]:
    if not categorical:
        st.info("No categorical columns found.")
    else:
        cols_per_row = 2
        for i in range(0, min(len(categorical), 8), cols_per_row):
            row_cols = st.columns(cols_per_row)
            for j, col in enumerate(categorical[i:i+cols_per_row]):
                with row_cols[j]:
                    vc = df[col].value_counts().head(15)
                    n_unique = df[col].nunique()
                    # Bar for high cardinality, pie for low
                    if n_unique <= 6:
                        fig = px.pie(
                            values=vc.values, names=vc.index,
                            color_discrete_sequence=PALETTE,
                        )
                        fig.update_traces(textfont_color="#f0eff5",
                                          textfont_size=10)
                    else:
                        fig = px.bar(
                            x=vc.values, y=vc.index,
                            orientation="h",
                            color_discrete_sequence=["#22d3ee"],
                            labels={"x": "Count", "y": col},
                        )
                        fig.update_layout(yaxis={"categoryorder": "total ascending"})
                    theme(fig, f"{col} ({n_unique} unique)")
                    st.plotly_chart(fig, use_container_width=True,
                                    config={"displayModeBar": False})

# ── TAB 4: Statistics ─────────────────────────────────────────
with tabs[3]:
    st.markdown('<div class="section-title">🔢 Numeric Summary Statistics</div>', unsafe_allow_html=True)
    if numeric:
        desc = df[numeric].describe().T.round(3)
        desc["skewness"] = df[numeric].skew().round(3)
        desc["kurtosis"] = df[numeric].kurtosis().round(3)
        desc["missing"] = df[numeric].isnull().sum()
        desc["missing_%"] = (df[numeric].isnull().mean() * 100).round(1)
        st.dataframe(desc, use_container_width=True)

    st.markdown('<div class="section-title">🗂️ Categorical Summary</div>', unsafe_allow_html=True)
    if categorical:
        cat_summary = pd.DataFrame({
            "Column": categorical,
            "Unique Values": [df[c].nunique() for c in categorical],
            "Most Common": [df[c].mode()[0] if not df[c].mode().empty else "N/A" for c in categorical],
            "Most Common %": [(df[c].value_counts(normalize=True).iloc[0]*100).round(1)
                              if df[c].notna().any() else 0 for c in categorical],
            "Missing": [df[c].isnull().sum() for c in categorical],
            "Missing %": [(df[c].isnull().mean()*100).round(1) for c in categorical],
        })
        st.dataframe(cat_summary, use_container_width=True, hide_index=True)

    # Missing value heatmap
    missing_any = df.isnull().sum()
    missing_any = missing_any[missing_any > 0]
    if not missing_any.empty:
        st.markdown('<div class="section-title">🕳️ Missing Value Map</div>', unsafe_allow_html=True)
        sample = df[missing_any.index].head(200).isnull().astype(int)
        fig_miss = px.imshow(
            sample.T,
            color_continuous_scale=[[0, "#111118"], [1, "#f87171"]],
            aspect="auto",
            labels={"color": "Missing"},
        )
        theme(fig_miss, "Missing Values (red = missing) — first 200 rows")
        fig_miss.update_layout(height=max(200, len(missing_any) * 30 + 60),
                                coloraxis_showscale=False)
        st.plotly_chart(fig_miss, use_container_width=True,
                        config={"displayModeBar": False})

# ── TAB 5: Raw Data ───────────────────────────────────────────
with tabs[4]:
    st.markdown('<div class="section-title">🧬 Raw Data Preview</div>', unsafe_allow_html=True)

    # Column types display
    type_html = ""
    for col in df.columns:
        dtype = str(df[col].dtype)
        type_html += f'<span class="col-chip">{col} <span style="color:#22d3ee">{dtype}</span></span>'
    st.markdown(type_html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Search/filter
    search = st.text_input("🔍 Filter columns", placeholder="Type column name...",
                            label_visibility="collapsed")
    if search:
        cols_show = [c for c in df.columns if search.lower() in c.lower()]
        if cols_show:
            st.dataframe(df[cols_show].head(500), use_container_width=True, height=400)
        else:
            st.warning("No columns match your search.")
    else:
        st.dataframe(df.head(500), use_container_width=True, height=400)

    # Download
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download full dataset as CSV",
        data=csv_bytes,
        file_name=f"{st.session_state.dataset_name or 'dataset'}_clean.csv",
        mime="text/csv",
    )

# ── Footer ────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; margin-top:3rem; padding-top:1.5rem;
     border-top:1px solid rgba(255,255,255,0.06); color:#444; font-size:0.78rem;">
  Built by <strong style="color:#666">Krishna Sanjay Vaddi</strong> ·
  <a href="https://github.com/KrishnaSanjay21" target="_blank"
     style="color:#22d3ee; text-decoration:none;">GitHub</a> ·
  <a href="https://www.linkedin.com/in/krishna-sanjay-vaddi" target="_blank"
     style="color:#22d3ee; text-decoration:none;">LinkedIn</a>
</div>
""", unsafe_allow_html=True)
