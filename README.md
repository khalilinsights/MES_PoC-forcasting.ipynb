
# MES Demand Forecasting — Proof of Concept

A Proof of Concept for **MetroEnergy Solutions (MES)**, a fictional UK/EU energy provider. This repository contains the Python notebook that demonstrates short-term electricity demand forecasting — the machine-learning component of a proposed cloud-based big data architecture.

---

## Overview

This PoC implements a Random Forest regressor that forecasts hourly electricity demand on a 30-day horizon. It is designed to exercise every layer of a proposed cloud architecture: data ingestion, lakehouse storage, batch feature engineering, ML training, and model serving.

**Key result:** The model achieves **1.84% MAPE** (770 MW MAE, 980 MW RMSE) on a 30-day held-out test set.

---

## Repository contents

| File | Description |
|---|---|
| `MES_PoC_Forecasting.ipynb` | Main Google Colab notebook — self-contained and reproducible |
| `mes_demand_data.csv` | Synthetic UK-style hourly demand dataset (17,520 rows, 2 years) |
| `build_poc.py` | Standalone Python script version of the notebook |
| `mes_architecture.png` | Architecture diagram |
| `poc_eda.png` | Exploratory data analysis |
| `poc_forecast.png` | 7-day forecast vs actual |
| `poc_feature_importance.png` | Feature importance bar chart |
| `README.md` | This file |

---

## How to run

### Option 1 — Google Colab (recommended)

1. Open `MES_PoC_Forecasting.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Select **Runtime → Run all**
3. The full notebook executes in under 2 minutes on a free CPU runtime
4. All three figures are saved to the Colab file browser (left sidebar)

No paid resources, GPUs, or external credentials are required.

### Option 2 — Local Python

```bash
pip install pandas numpy scikit-learn matplotlib
python build_poc.py
```

Requires Python 3.9 or later.

---

## The dataset

The dataset (`mes_demand_data.csv`) is **synthetic** and generated programmatically within the notebook. Real UK smart-meter data is not freely available due to GDPR, so this PoC uses a synthetic dataset modelled on documented UK National Grid consumption patterns.

**17,520 hourly observations** spanning 2023-01-01 to 2024-12-31, with six columns:

| Column | Description |
|---|---|
| `timestamp` | Hourly datetime |
| `demand_mw` | Electricity demand (megawatts) |
| `temperature_c` | Temperature (degrees Celsius) |
| `hour` | Hour of day (0–23) |
| `dayofweek` | Day of week (0=Monday) |
| `month` | Month (1–12) |

**Four signals** built into the generator to mirror real UK demand:

- Diurnal cycle — peak around 6pm
- Weekly cycle — weekend dip
- Seasonal cycle — winter peaks, summer troughs
- Temperature sensitivity — cold weather raises demand

A fixed random seed (`numpy.random.seed(42)`) makes the dataset and results **exactly reproducible**.

---

## Notebook structure

The notebook follows a clean data-science workflow:

| Section | What it does |
|---|---|
| **Data Importing** | Generates the synthetic 2-year hourly demand series |
| **Data Cleaning** | Drops rows with undefined lag values |
| **Data Processing** | Creates lag features and chronological train/test split |
| **The Model** | Trains Random Forest (120 trees, depth 18) |
| **Visualization** | Produces EDA, forecast, and feature importance charts |
| **Results** | Prints final MAE, MAPE, RMSE metrics |

---

## Model details

- **Algorithm:** Random Forest Regressor (scikit-learn)
- **Hyperparameters:** 120 trees, maximum depth 18, random state 42
- **Features (7):** hour, dayofweek, month, temperature_c, lag_1h, lag_24h, lag_168h
- **Train/test split:** Chronological — first 23 months for training, final 30 days held out
- **Evaluation metrics:** MAE, MAPE, RMSE

---

## Results

| Metric | Value |
|---|---|
| MAE | ~770 MW |
| MAPE | **1.84%** |
| RMSE | ~980 MW |

**Feature importance ranking (top 3):**

1. `lag_1h` (short-term momentum) — ~70% importance
2. `lag_168h` (weekly cycle) — ~21% importance
3. `lag_24h` (daily cycle) — ~6% importance

Temperature contributes meaningfully (~1.4%), validating the inclusion of weather data in the ingestion design. Calendar features (hour, day-of-week, month) are largely subsumed by lag features but remain useful for cold-start prediction.

---

## Architecture connection

This PoC implements the **machine-learning engine of the processing layer** in the proposed MES architecture. In production, the synthetic data generator would be replaced by a pandas read of Gold-layer Parquet files from a lakehouse via Amazon Athena or an equivalent SQL engine, with the model retrained nightly and predictions served through a private REST API.

See the full architecture in `mes_architecture.png`.

---

## Technologies used

- **Python 3** — language
- **pandas** — data handling
- **NumPy** — numerical operations
- **scikit-learn** — Random Forest and evaluation metrics
- **matplotlib** — visualisation
- **Google Colab** — execution environment

---

## Limitations

- Synthetic dataset does not reflect special events (public holidays, televised broadcasts, industrial action)
- Random Forest is a strong baseline; in production, LightGBM and Temporal Fusion Transformer would also be benchmarked
- The 30-day evaluation window does not exercise a full annual cycle — rolling-origin cross-validation would be the next analytical step
