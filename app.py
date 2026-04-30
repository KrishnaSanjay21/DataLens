import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="DataLens — EDA Report Generator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem !important; max-width: 100% !important; }
.app-header { display:flex; align-items:center; gap:1rem; margin-bottom:1.75rem; padding-bottom:1.25rem; border-bottom:1px solid rgba(255,255,255,0.07); }
.app-title { font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800; color:#f0eff5; letter-spacing:-0.025em; margin:0; }
.app-title span { color:#22d3ee; }
.app-badge { font-size:0.68rem; font-weight:500; letter-spacing:0.1em; text-transform:uppercase; background:rgba(34,211,238,0.1); color:#22d3ee; border:1px solid rgba(34,211,238,0.25); padding:3px 10px; border-radius:999px; }
.stat-card { background:#111118; border:1px solid rgba(255,255,255,0.07); border-radius:14px; padding:1rem 1.25rem; height:100%; }
.stat-label { font-size:0.68rem; text-transform:uppercase; letter-spacing:0.1em; color:#666; margin-bottom:4px; }
.stat-value { font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800; color:#22d3ee; letter-spacing:-0.02em; line-height:1.1; }
.stat-sub { font-size:0.72rem; color:#555; margin-top:3px; }
.section-title { font-family:'Syne',sans-serif; font-size:1.05rem; font-weight:700; color:#f0eff5; margin:2rem 0 1rem; display:flex; align-items:center; gap:8px; }
.section-title::before { content:''; display:inline-block; width:3px; height:16px; background:#22d3ee; border-radius:2px; }
.badge-warn { display:inline-block; background:rgba(251,191,36,0.1); color:#fbbf24; border:1px solid rgba(251,191,36,0.25); padding:2px 10px; border-radius:999px; font-size:0.72rem; font-weight:500; margin:2px; }
.badge-ok { display:inline-block; background:rgba(52,211,153,0.1); color:#34d399; border:1px solid rgba(52,211,153,0.25); padding:2px 10px; border-radius:999px; font-size:0.72rem; font-weight:500; margin:2px; }
.badge-err { display:inline-block; background:rgba(239,68,68,0.1); color:#f87171; border:1px solid rgba(239,68,68,0.25); padding:2px 10px; border-radius:999px; font-size:0.72rem; font-weight:500; margin:2px; }
.col-chip { display:inline-block; background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.09); border-radius:6px; padding:2px 8px; font-size:0.7rem; color:#888; margin:2px; }
.insight-box { background:rgba(34,211,238,0.05); border:1px solid rgba(34,211,238,0.15); border-left:3px solid #22d3ee; border-radius:0 10px 10px 0; padding:0.75rem 1rem; font-size:0.85rem; color:#c8c7d4; line-height:1.65; margin:0.5rem 0; }
.insight-box strong { color:#f0eff5; }
.stButton > button { background:#22d3ee !important; color:#0a0a0f !important; border:none !important; border-radius:10px !important; font-weight:600 !important; font-size:0.875rem !important; padding:0.6rem 1.5rem !important; }
.stButton > button:hover { background:#06b6d4 !important; transform:translateY(-1px) !important; }
div[data-testid="stRadio"] label { font-size:0.875rem !important; }
.stTabs [data-baseweb="tab"] { font-size:0.82rem !important; font-weight:500 !important; color:#666 !important; }
.stTabs [aria-selected="true"] { color:#22d3ee !important; }
::-webkit-scrollbar { width:4px; height:4px; }
::-webkit-scrollbar-thumb { background:rgba(255,255,255,0.1); border-radius:2px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────
BG = "#0a0a0f"
SURFACE = "#111118"
GRID = "rgba(255,255,255,0.05)"
TEXT = "#888797"
PALETTE = ["#22d3ee","#fbbf24","#a78bfa","#34d399","#f472b6","#fb923c","#60a5fa","#e879f9"]
MAX_ROWS_CHART = 5000   # sample for heavy charts
MAX_COLS_SCATTER = 5    # scatter matrix column limit

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

# ── Demo loaders — cached so they only run once ───────────────
@st.cache_data(show_spinner=False)
def load_titanic():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    try:
        return pd.read_csv(url)
    except:
        np.random.seed(42)
        n = 891
        return pd.DataFrame({
            "Survived": np.random.choice([0,1], n, p=[0.62,0.38]),
            "Pclass": np.random.choice([1,2,3], n, p=[0.24,0.21,0.55]),
            "Sex": np.random.choice(["male","female"], n, p=[0.65,0.35]),
            "Age": np.where(np.random.rand(n)<0.2, np.nan, np.random.normal(30,14,n).clip(1,80)),
            "SibSp": np.random.choice(range(6), n),
            "Fare": np.abs(np.random.lognormal(3.2,1.0,n)).round(2),
            "Embarked": np.random.choice(["S","C","Q",None], n, p=[0.72,0.19,0.086,0.004]),
        })

@st.cache_data(show_spinner=False)
def load_iris():
    try:
        from sklearn.datasets import load_iris as _li
        d = _li()
        df = pd.DataFrame(d.data, columns=d.feature_names)
        df["species"] = pd.Categorical.from_codes(d.target, d.target_names)
        return df
    except:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_nyc():
    np.random.seed(99)
    n = 5000
    boroughs = ["Manhattan","Brooklyn","Queens","Bronx","Staten Island"]
    return pd.DataFrame({
        "pickup_hour": np.random.choice(range(24), n),
        "pickup_day": np.random.choice(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], n),
        "pickup_borough": np.random.choice(boroughs, n, p=[0.55,0.20,0.15,0.07,0.03]),
        "dropoff_borough": np.random.choice(boroughs, n, p=[0.55,0.20,0.15,0.07,0.03]),
        "passenger_count": np.random.choice(range(1,7), n),
        "trip_distance_miles": np.abs(np.random.lognormal(0.9,0.8,n)).clip(0.1,35).round(2),
        "fare_amount": np.abs(np.random.normal(14,8,n)).clip(2.5,120).round(2),
        "tip_amount": np.abs(np.random.normal(2.5,2,n)).clip(0,30).round(2),
        "payment_type": np.random.choice(["Credit Card","Cash","No Charge"], n, p=[0.67,0.31,0.02]),
        "speed_mph": np.abs(np.random.normal(18,8,n)).clip(1,60).round(1),
    })

# ── All heavy analysis cached by dataframe hash ───────────────
@st.cache_data(show_spinner=False)
def analyse(df: pd.DataFrame):
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=["object","category","bool"]).columns.tolist()

    # Issues
    issues = []
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(1)
    for col in df.columns:
        if missing[col] > 0:
            sev = "high" if missing_pct[col] > 30 else "medium" if missing_pct[col] > 5 else "low"
            issues.append({"column": col, "issue": "Missing values",
                           "detail": f"{missing[col]:,} missing ({missing_pct[col]}%)", "severity": sev})
    dupes = int(df.duplicated().sum())
    if dupes > 0:
        sev = "high" if dupes/len(df) > 0.05 else "medium"
        issues.append({"column": "All", "issue": "Duplicate rows",
                       "detail": f"{dupes:,} duplicate rows ({dupes/len(df)*100:.1f}%)", "severity": sev})
    for col in numeric:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        out = int(((df[col] < Q1-3*IQR) | (df[col] > Q3+3*IQR)).sum())
        if out > 0:
            sev = "medium" if out/len(df) > 0.01 else "low"
            issues.append({"column": col, "issue": "Outliers (3×IQR)",
                           "detail": f"{out:,} extreme outliers ({out/len(df)*100:.1f}%)", "severity": sev})
    for col in categorical:
        if df[col].nunique() > 50:
            issues.append({"column": col, "issue": "High cardinality",
                           "detail": f"{df[col].nunique():,} unique values", "severity": "low"})
    for col in df.columns:
        if df[col].nunique() <= 1:
            issues.append({"column": col, "issue": "Constant column",
                           "detail": "Only 1 unique value", "severity": "high"})
    for col in numeric:
        skew = float(df[col].skew())
        if abs(skew) > 2:
            issues.append({"column": col, "issue": "High skewness",
                           "detail": f"Skewness={skew:.2f} — consider log transform", "severity": "low"})

    # Insights
    insights = []
    insights.append(f"Dataset has <strong>{len(df):,} rows</strong> and <strong>{len(df.columns)} columns</strong> — {len(numeric)} numeric, {len(categorical)} categorical.")
    total_missing = int(df.isnull().sum().sum())
    if total_missing > 0:
        insights.append(f"<strong>{total_missing:,} missing values</strong> ({total_missing/df.size*100:.1f}% of all cells). Columns with >30% missing may need dropping.")
    else:
        insights.append("✅ <strong>No missing values</strong> — clean dataset!")
    for col in numeric[:3]:
        skew = float(df[col].skew())
        if abs(skew) > 1:
            d = "right (positive)" if skew > 0 else "left (negative)"
            insights.append(f"<strong>{col}</strong> is {d}-skewed (skewness={skew:.2f}). Consider log/sqrt transform for ML.")
    if len(numeric) >= 2:
        corr_m = df[numeric].corr().abs()
        _corr_vals = corr_m.values.copy()
        np.fill_diagonal(_corr_vals, 0)
        corr_m = pd.DataFrame(_corr_vals, index=corr_m.index, columns=corr_m.columns)
        mx = float(corr_m.max().max())
        if mx > 0.8:
            idx = corr_m.stack().idxmax()
            insights.append(f"Strong correlation between <strong>{idx[0]}</strong> and <strong>{idx[1]}</strong> (r={mx:.2f}). Risk of multicollinearity.")
    for col in categorical[:2]:
        vc = df[col].value_counts(normalize=True)
        if len(vc) > 0 and vc.iloc[0] > 0.85:
            insights.append(f"<strong>{col}</strong> is imbalanced — '{vc.index[0]}' = {vc.iloc[0]*100:.0f}% of records.")

    # Stats tables
    desc = df[numeric].describe().T.round(3) if numeric else pd.DataFrame()
    if not desc.empty:
        desc["skewness"] = df[numeric].skew().round(3)
        desc["kurtosis"] = df[numeric].kurtosis().round(3)
        desc["missing"] = df[numeric].isnull().sum()
        desc["missing_%"] = (df[numeric].isnull().mean()*100).round(1)

    cat_summary = pd.DataFrame()
    if categorical:
        cat_summary = pd.DataFrame({
            "Column": categorical,
            "Unique": [df[c].nunique() for c in categorical],
            "Most Common": [str(df[c].mode()[0]) if df[c].notna().any() else "N/A" for c in categorical],
            "Most Common %": [(df[c].value_counts(normalize=True).iloc[0]*100).round(1)
                              if df[c].notna().any() else 0 for c in categorical],
            "Missing": [df[c].isnull().sum() for c in categorical],
            "Missing %": [(df[c].isnull().mean()*100).round(1) for c in categorical],
        })

    return numeric, categorical, issues, insights, desc, cat_summary

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <span style="font-size:1.75rem">🔬</span>
  <div><div class="app-title">Data<span>Lens</span></div></div>
  <span class="app-badge">EDA Report Generator</span>
  <span style="margin-left:auto;font-size:0.75rem;color:#444;">
    Instant profiling · Zero setup · No API key
  </span>
</div>
""", unsafe_allow_html=True)

# ── Data source ───────────────────────────────────────────────
st.markdown("#### 📂 Load your data")
source = st.radio("Source", ["🗂️ Upload CSV","🚢 Titanic","🌸 Iris","🚕 NYC Taxi"],
                  horizontal=True, label_visibility="collapsed")

df = None
dataset_name = ""

if source == "🗂️ Upload CSV":
    uploaded = st.file_uploader("Drop a CSV file", type=["csv"], label_visibility="collapsed")
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            dataset_name = uploaded.name
            st.success(f"✅ Loaded **{uploaded.name}** — {len(df):,} rows × {len(df.columns)} cols")
        except Exception as e:
            st.error(f"Could not read file: {e}")
else:
    demo_map = {
        "🚢 Titanic": ("Titanic — Passenger Survival", load_titanic),
        "🌸 Iris": ("Iris — Flower Measurements", load_iris),
        "🚕 NYC Taxi": ("NYC Taxi Trips", load_nyc),
    }
    name, loader = demo_map[source]
    dataset_name = name
    with st.spinner(f"Loading {name}..."):
        df = loader()
    if df is not None and not df.empty:
        st.success(f"✅ Loaded **{name}** — {len(df):,} rows × {len(df.columns)} cols")

# ── Guard ─────────────────────────────────────────────────────
if df is None or df.empty:
    st.markdown("""
    <div style="text-align:center;padding:4rem 0;color:#444;">
        <div style="font-size:3rem;margin-bottom:1rem">🔬</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.1rem;color:#666;margin-bottom:0.5rem">
            No data loaded yet
        </div>
        <div style="font-size:0.85rem">Upload a CSV or pick a demo dataset above</div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── Run all analysis (cached) ─────────────────────────────────
with st.spinner("Analysing dataset..."):
    numeric, categorical, issues, insights, desc, cat_summary = analyse(df)

total_missing = int(df.isnull().sum().sum())
missing_pct_total = round(total_missing / df.size * 100, 1)
dupes = int(df.duplicated().sum())
mem_mb = round(df.memory_usage(deep=True).sum() / 1024**2, 2)

# ── Overview metrics ──────────────────────────────────────────
st.markdown(f'<div class="section-title">📊 Overview — {dataset_name}</div>', unsafe_allow_html=True)
m1,m2,m3,m4,m5,m6 = st.columns(6)
for col, label, value, sub, color in [
    (m1, "Rows",       f"{len(df):,}",           "records",                    "#22d3ee"),
    (m2, "Columns",    str(len(df.columns)),      f"{len(numeric)} num · {len(categorical)} cat", "#22d3ee"),
    (m3, "Missing",    f"{missing_pct_total}%",   f"{total_missing:,} cells",   "#f87171" if missing_pct_total>10 else "#22d3ee"),
    (m4, "Duplicates", f"{dupes:,}",              f"{dupes/len(df)*100:.1f}% of rows", "#f87171" if dupes>0 else "#22d3ee"),
    (m5, "Issues",     str(len(issues)),          "detected",                   "#fbbf24" if issues else "#34d399"),
    (m6, "Memory",     f"{mem_mb} MB",            "in memory",                  "#22d3ee"),
]:
    with col:
        st.markdown(f'<div class="stat-card"><div class="stat-label">{label}</div>'
                    f'<div class="stat-value" style="color:{color}">{value}</div>'
                    f'<div class="stat-sub">{sub}</div></div>', unsafe_allow_html=True)

# ── Insights ──────────────────────────────────────────────────
st.markdown('<div class="section-title">💡 Auto Insights</div>', unsafe_allow_html=True)
for ins in insights:
    st.markdown(f'<div class="insight-box">{ins}</div>', unsafe_allow_html=True)

# ── Issues ────────────────────────────────────────────────────
if issues:
    st.markdown('<div class="section-title">⚠️ Data Quality Issues</div>', unsafe_allow_html=True)
    import pandas as _pd
    idf = _pd.DataFrame(issues)
    for sev, badge, emoji in [("high","badge-err","🔴"),("medium","badge-warn","🟡"),("low","badge-ok","🟢")]:
        grp = idf[idf.severity == sev]
        if not grp.empty:
            st.markdown(f"**{emoji} {sev.title()} severity**")
            for _, row in grp.iterrows():
                st.markdown(f'<span class="{badge}">{row["column"]}</span> '
                            f'<span style="font-size:0.82rem;color:#888">{row["issue"]} — {row["detail"]}</span>',
                            unsafe_allow_html=True)
            st.markdown("")
else:
    st.markdown('<div class="insight-box">✅ <strong>No data quality issues!</strong></div>', unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────
tabs = st.tabs(["📈 Distributions", "🔗 Correlations", "🗂️ Categorical", "🔢 Statistics", "🧬 Raw Data"])

# sample for heavy chart renders
plot_df = df.sample(min(MAX_ROWS_CHART, len(df)), random_state=42) if len(df) > MAX_ROWS_CHART else df

# TAB 1 — Distributions
with tabs[0]:
    if not numeric:
        st.info("No numeric columns found.")
    else:
        n_cols = 3
        for i in range(0, len(numeric), n_cols):
            row_cols = st.columns(n_cols)
            for j, col in enumerate(numeric[i:i+n_cols]):
                with row_cols[j]:
                    s = plot_df[col].dropna()
                    skew = float(s.skew())
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=s, nbinsx=25,
                                               marker_color="#22d3ee", opacity=0.85, name=col))
                    fig.add_vline(x=float(s.mean()), line_dash="dash",
                                  line_color="#fbbf24", line_width=1.5,
                                  annotation_text="mean", annotation_font_size=9)
                    fig.add_vline(x=float(s.median()), line_dash="dot",
                                  line_color="#a78bfa", line_width=1.5,
                                  annotation_text="median", annotation_font_size=9)
                    theme(fig, f"{col} (skew={skew:.2f})")
                    fig.update_layout(height=240, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="section-title">📦 Box Plots</div>', unsafe_allow_html=True)
        fig_box = go.Figure()
        for i, col in enumerate(numeric[:8]):
            fig_box.add_trace(go.Box(y=plot_df[col].dropna(), name=col,
                                     marker_color=PALETTE[i % len(PALETTE)],
                                     boxmean=True, line_width=1.5))
        theme(fig_box, "Box Plots — Numeric Columns")
        fig_box.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True, config={"displayModeBar": False})

# TAB 2 — Correlations
with tabs[1]:
    if len(numeric) < 2:
        st.info("Need at least 2 numeric columns.")
    else:
        corr = df[numeric].corr()
        fig_heat = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
            colorscale=[[0,"#1e0a3c"],[0.5,"#111118"],[1,"#22d3ee"]],
            zmid=0, text=corr.round(2).values,
            texttemplate="%{text}", textfont={"size": 9}, showscale=True,
        ))
        theme(fig_heat, "Correlation Matrix")
        fig_heat.update_layout(height=max(320, len(numeric)*42))
        st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})

        # Ranked table
        st.markdown('<div class="section-title">🔝 Strongest Correlations</div>', unsafe_allow_html=True)
        corr_pairs = (corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                      .stack().reset_index())
        corr_pairs.columns = ["Feature A","Feature B","Correlation"]
        corr_pairs["Abs"] = corr_pairs["Correlation"].abs()
        corr_pairs = corr_pairs.sort_values("Abs", ascending=False).head(10).drop("Abs", axis=1)
        corr_pairs["Correlation"] = corr_pairs["Correlation"].round(3)
        corr_pairs["Strength"] = corr_pairs["Correlation"].abs().apply(
            lambda x: "🔴 Strong" if x > 0.7 else "🟡 Moderate" if x > 0.4 else "🟢 Weak")
        st.dataframe(corr_pairs, use_container_width=True, hide_index=True)

        # Scatter matrix — capped columns + sampled rows
        top_feats = numeric[:MAX_COLS_SCATTER]
        color_col = categorical[0] if categorical else None
        sample_size = min(1000, len(df))
        sdf = df[top_feats + ([color_col] if color_col else [])].dropna().sample(sample_size, random_state=1)
        st.markdown('<div class="section-title">🔵 Scatter Matrix (sampled)</div>', unsafe_allow_html=True)
        fig_sc = px.scatter_matrix(sdf, dimensions=top_feats, color=color_col,
                                   color_discrete_sequence=PALETTE, opacity=0.45)
        fig_sc.update_traces(marker=dict(size=2))
        theme(fig_sc)
        fig_sc.update_layout(height=460)
        st.plotly_chart(fig_sc, use_container_width=True, config={"displayModeBar": False})

# TAB 3 — Categorical
with tabs[2]:
    if not categorical:
        st.info("No categorical columns found.")
    else:
        for i in range(0, min(len(categorical), 8), 2):
            row_cols = st.columns(2)
            for j, col in enumerate(categorical[i:i+2]):
                with row_cols[j]:
                    vc = df[col].value_counts().head(12)
                    n_unique = df[col].nunique()
                    if n_unique <= 6:
                        fig = px.pie(values=vc.values, names=vc.index,
                                     color_discrete_sequence=PALETTE)
                        fig.update_traces(textfont_color="#f0eff5", textfont_size=10)
                    else:
                        fig = px.bar(x=vc.values, y=vc.index, orientation="h",
                                     color_discrete_sequence=["#22d3ee"],
                                     labels={"x":"Count","y":col})
                        fig.update_layout(yaxis={"categoryorder":"total ascending"})
                    theme(fig, f"{col} ({n_unique} unique)")
                    fig.update_layout(height=260)
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# TAB 4 — Statistics
with tabs[3]:
    if not desc.empty:
        st.markdown('<div class="section-title">🔢 Numeric Summary</div>', unsafe_allow_html=True)
        st.dataframe(desc, use_container_width=True)
    if not cat_summary.empty:
        st.markdown('<div class="section-title">🗂️ Categorical Summary</div>', unsafe_allow_html=True)
        st.dataframe(cat_summary, use_container_width=True, hide_index=True)
    # Missing heatmap
    missing_cols = [c for c in df.columns if df[c].isnull().any()]
    if missing_cols:
        st.markdown('<div class="section-title">🕳️ Missing Value Map</div>', unsafe_allow_html=True)
        sample = df[missing_cols].head(150).isnull().astype(int)
        fig_miss = px.imshow(sample.T,
                             color_continuous_scale=[[0,"#111118"],[1,"#f87171"]],
                             aspect="auto", labels={"color":"Missing"})
        theme(fig_miss, "Missing Values (red = missing) — first 150 rows")
        fig_miss.update_layout(height=max(180, len(missing_cols)*28+60), coloraxis_showscale=False)
        st.plotly_chart(fig_miss, use_container_width=True, config={"displayModeBar": False})

# TAB 5 — Raw Data
with tabs[4]:
    type_html = "".join(
        f'<span class="col-chip">{c} <span style="color:#22d3ee">{str(df[c].dtype)}</span></span>'
        for c in df.columns
    )
    st.markdown(type_html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    search = st.text_input("🔍 Filter columns", placeholder="Type column name...",
                           label_visibility="collapsed")
    show_cols = [c for c in df.columns if search.lower() in c.lower()] if search else df.columns.tolist()
    st.dataframe(df[show_cols].head(300), use_container_width=True, height=380)
    st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name=f"{dataset_name}_export.csv", mime="text/csv")

st.markdown("""
<div style="text-align:center;margin-top:3rem;padding-top:1.5rem;
     border-top:1px solid rgba(255,255,255,0.06);color:#444;font-size:0.78rem;">
  Built by <strong style="color:#666">Krishna Sanjay Vaddi</strong> ·
  <a href="https://github.com/KrishnaSanjay21" target="_blank" style="color:#22d3ee;text-decoration:none;">GitHub</a> ·
  <a href="https://www.linkedin.com/in/krishna-sanjay-vaddi" target="_blank" style="color:#22d3ee;text-decoration:none;">LinkedIn</a>
</div>
""", unsafe_allow_html=True)