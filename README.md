<div align="center">

# 🔬 DataLens — Automated EDA Report Generator

**Upload any CSV. Get a full data quality report, charts, and insights — instantly.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![No API Key](https://img.shields.io/badge/API%20Key-Not%20Required-34d399?style=flat)


[Live Demo](https://datalens-x4seve4uqecrujmqwmzoth.streamlit.app/) · [Report Bug](https://github.com/KrishnaSanjay21/datalens/issues) · [LinkedIn](https://www.linkedin.com/in/krishna-sanjay-vaddi)

</div>

---

## 📌 Overview

DataLens is a zero-config, no-API-key EDA tool. Drop in any CSV and instantly get:
- Data quality issues flagged by severity
- Auto-generated plain-English insights
- Distribution histograms with mean/median overlays
- Correlation heatmap + scatter matrix
- Categorical breakdowns
- Full summary statistics with skewness & kurtosis
- Missing value heatmap

> No API key. No setup. Just upload and explore.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📂 **CSV Upload** | Upload any CSV for instant profiling |
| 🗂️ **Demo Datasets** | Titanic, Iris, NYC Taxi built in |
| ⚠️ **Issue Detection** | Missing values, duplicates, outliers, skewness, high cardinality |
| 💡 **Auto Insights** | Plain-English summaries generated from statistics |
| 📈 **Distributions** | Histograms with mean/median lines per numeric column |
| 📦 **Box Plots** | Side-by-side outlier view across all numeric columns |
| 🔗 **Correlations** | Heatmap + scatter matrix + ranked correlation table |
| 🗂️ **Categorical** | Bar/pie charts per categorical column |
| 🔢 **Statistics** | Full describe() + skewness, kurtosis, missing% |
| 🧬 **Raw Data** | Searchable preview with column filter + CSV download |

---

## 🚀 Deploy to Streamlit Cloud

### Step 1 — Push to GitHub
```bash
git init
git add .
git commit -m "🔬 Initial commit — DataLens EDA"
git remote add origin https://github.com/KrishnaSanjay21/datalens.git
git push -u origin main
```

### Step 2 — Deploy
1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Click **"Create app"** → connect your GitHub repo
3. Set **Main file path** to `app.py`
4. Click **Deploy** ✅ ← No secrets needed!

---

## 💻 Run Locally

```bash
git clone https://github.com/KrishnaSanjay21/datalens.git
cd datalens
pip install -r requirements.txt
streamlit run app.py
```

---

## 📊 What Gets Analyzed

### Data Quality Checks
- 🔴 **High severity** — constant columns, >30% missing, >5% duplicates
- 🟡 **Medium severity** — 5–30% missing, outliers >1% of data
- 🟢 **Low severity** — high cardinality, mild skewness, minor outliers

### Auto Insights Cover
- Dataset shape and column type breakdown
- Missing data percentage and recommendations
- Skewness detection with transform suggestions
- Strong correlation warnings for multicollinearity
- Class imbalance detection

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **UI** | Streamlit + Custom CSS |
| **Charts** | Plotly Express + Plotly Graph Objects |
| **Data Processing** | Pandas + NumPy |
| **ML Utilities** | Scikit-learn (Iris dataset) |
| **Deployment** | Streamlit Cloud |

> ✅ No LLM / API key required — fully statistical, 100% free to run.

---

## 📁 Project Structure

```
datalens/
├── app.py                  # Main Streamlit application
├── requirements.txt        # 5 dependencies, no API keys
├── .gitignore
├── .streamlit/
│   └── config.toml         # Dark theme
└── README.md
```

---

## 👤 Author

**Krishna Sanjay Vaddi**
MS Data Science & Analytics @ Florida Atlantic University

[![GitHub](https://img.shields.io/badge/GitHub-KrishnaSanjay21-181717?style=flat&logo=github)](https://github.com/KrishnaSanjay21)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-krishna--sanjay--vaddi-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/krishna-sanjay-vaddi)

---

<div align="center">
  <sub>Built with 🔬 + Streamlit + Plotly</sub>
</div>
