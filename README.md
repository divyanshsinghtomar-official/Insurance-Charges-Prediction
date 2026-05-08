<div align="center">

# Insurance Charges Predictor

### *A Machine Learning Pipeline for Medical Cost Prediction*

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Scientific-013243?style=for-the-badge&logo=numpy&logoColor=white)

<br/>

![Status](https://img.shields.io/badge/Status-Complete-2ecc71?style=flat-square)
![Model](https://img.shields.io/badge/Model-Linear%20Regression-9b59b6?style=flat-square)
![R²](https://img.shields.io/badge/R²%20Score-High%20Accuracy-e74c3c?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)

<br/>

> **Predict medical insurance charges** using demographic and lifestyle features — powered by feature engineering, interaction terms, and regularised regression.

<br/>

---

</div>

## 📌 Table of Contents

- [📖 Overview](#-overview)
- [✨ Features](#-features)
- [📊 Dataset](#-dataset)
- [🔬 Methodology](#-methodology)
- [🧠 Feature Engineering](#-feature-engineering)
- [📈 Model Performance](#-model-performance)
- [🛠️ Tech Stack](#️-tech-stack)
- [🚀 Getting Started](#-getting-started)
- [📁 Project Structure](#-project-structure)

---

## 📖 Overview

This project builds an end-to-end machine learning pipeline to **predict individual medical insurance charges** based on patient demographics and lifestyle attributes. It covers the full data science workflow — from raw data exploration all the way through to a tuned, regularised regression model with meaningful interaction features.

The key insight driving accuracy: **smoking status interacts non-linearly with age and BMI**, creating distinct cost clusters that a naive linear model misses entirely.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Exploratory Data Analysis** | Distribution plots, count plots, box plots, and correlation heatmaps |
| 🧹 **Data Cleaning** | Duplicate removal, null checks, dtype inspection |
| 🔢 **Encoding** | Binary mapping for `sex` / `smoker`; one-hot encoding for `region` |
| ⚗️ **Feature Engineering** | Interaction terms, polynomial features, BMI risk flags, log-target transform |
| 📐 **Feature Selection** | Pearson correlation + Chi-squared statistical tests |
| 📏 **Scaling** | `StandardScaler` on continuous features |
| 🤖 **Modelling** | `LinearRegression` with engineered feature set |
| 📊 **Evaluation** | R² score + Adjusted R² score |

---

## 📊 Dataset

**File:** `insurance.csv`

The dataset contains **1,338 records** of health insurance beneficiaries in the United States.

| Column | Type | Description |
|---|---|---|
| `age` | Numerical | Age of the primary beneficiary |
| `sex` | Categorical | Gender: `male` / `female` |
| `bmi` | Numerical | Body Mass Index (kg/m²) |
| `children` | Numerical | Number of dependents covered |
| `smoker` | Categorical | Smoking status: `yes` / `no` |
| `region` | Categorical | US residential region (4 zones) |
| `charges` | Numerical ⭐ | **Target** — Individual medical costs billed by health insurance |

> 💡 **Key Insight:** Smoking status is the single most powerful predictor of insurance charges, creating a bimodal cost distribution.

---

## 🔬 Methodology

The pipeline follows a structured 6-stage workflow:

```
📥 Raw Data
    │
    ▼
🔍 Exploratory Data Analysis
    │  • Distribution plots (histplot + KDE)
    │  • Count plots for categorical features
    │  • Box plots for outlier detection
    │  • Correlation heatmap
    ▼
🧹 Data Cleaning & Preprocessing
    │  • Drop duplicate rows
    │  • Binary encode sex & smoker
    │  • One-hot encode region (drop_first=True)
    ▼
⚗️ Feature Engineering & Scaling
    │  • BMI categories (underweight / normal / overweight / obese)
    │  • Interaction terms: smoker×age, smoker×bmi, age×bmi
    │  • Polynomial terms: age², bmi²
    │  • Log-transform target: log1p(charges)
    │  • StandardScaler on continuous columns
    ▼
📐 Feature Selection
    │  • Pearson correlation with target
    │  • Chi-squared test (α = 0.05)
    ▼
🤖 Model Training & Evaluation
       • Train/Test split (67% / 33%)
       • LinearRegression fit
       • R² + Adjusted R² scoring
```

---

## 🧠 Feature Engineering

This is where the model earns its accuracy. Seven new features are derived from the original columns:

| New Feature | Formula | Why It Matters |
|---|---|---|
| `smoker_age` | `issmoker × age` | Smokers' costs compound with age — a purely additive model misses this |
| `smoker_bmi` | `issmoker × bmi` | High-BMI smokers form a distinct, high-cost cluster |
| `age_bmi` | `age × bmi` | Joint effect of ageing with higher body mass |
| `age_sq` | `age²` | Healthcare costs grow non-linearly with age |
| `bmi_sq` | `bmi²` | Non-linear cost growth at extreme BMI values |
| `bmi_obese` | `bmi ≥ 30 → 1` | Binary obesity flag — strong threshold effect |
| `smoker_obese` | `issmoker × bmi_obese` | The highest-risk combination in the dataset |
| `log_charges` | `log1p(charges)` | Stabilises right-skewed target for better model assumptions |

> ⚠️ **BMI Categories Created:** `underweight` (< 18.5) · `normal` (18.5–24.9) · `overweight` (25–29.9) · `obese` (≥ 30)

---

## 📈 Model Performance

The model is evaluated using two metrics:

| Metric | Formula | Purpose |
|---|---|---|
| **R² Score** | `1 − SS_res/SS_tot` | Overall variance explained |
| **Adjusted R²** | `1 − (1−R²)(n−1)/(n−p−1)` | Penalises unnecessary features |

```python
# Evaluation snippet
r2     = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
```

> 📌 Adjusted R² is preferred here because the feature set was expanded with engineered columns — it guards against spurious R² inflation.

---

## 🛠️ Tech Stack

```
🐍  Python 3.10+
📓  Jupyter Notebook
🐼  Pandas          — data manipulation
🔢  NumPy           — numerical computing
📊  Matplotlib      — base plotting
🎨  Seaborn         — statistical visualisation
🔬  SciPy           — statistical tests (Pearson, Chi²)
🤖  scikit-learn    — preprocessing, modelling, evaluation
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn jupyter
```

### Run the Notebook

```bash
# 1. Clone the repository
git clone https://github.com/your-username/insurance-charges-predictor.git
cd insurance-charges-predictor

# 2. Make sure insurance.csv is in the root directory

# 3. Launch Jupyter
jupyter notebook main.ipynb
```

### Quick Start with pip environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook main.ipynb
```

---

## 📁 Project Structure

```
insurance-charges-predictor/
│
├── 📓 main.ipynb           ← Main notebook (EDA → Model)
├── 📄 insurance.csv        ← Raw dataset
├── 📋 README.md            ← You are here
└── 📦 requirements.txt     ← Python dependencies
```

---

## 🔄 Pipeline at a Glance

```
insurance.csv
    └── EDA (distributions · heatmaps · box plots)
         └── Cleaning (dedup · encode · rename)
              └── Feature Engineering (interactions · poly · log)
                   └── Selection (Pearson r · Chi² test)
                        └── Scaling (StandardScaler)
                             └── LinearRegression → R² · Adjusted R²
```

---

<div align="center">

### 🌟 If this project helped you, give it a star!

Made with ❤️ using Python & scikit-learn

![Python](https://img.shields.io/badge/Built%20with-Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Powered%20by-Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

</div>
