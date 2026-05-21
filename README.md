# Weather Forecasting System — Astana, Kazakhstan
## Final Project | Machine Learning (Advanced Level)
### Astana IT University

---

## Project Overview

Multi-model system for next-day temperature forecasting using 10 years of daily weather data (2015–2024).

**5 Models implemented:**
| # | Model | Type |
|---|-------|------|
| 1 | Linear Regression | Classical ML (baseline) |
| 2 | Random Forest | Ensemble / Tree-based |
| 3 | XGBoost | Gradient Boosting |
| 4 | GRU | Deep Learning / RNN |
| 5 | LSTM | Deep Learning / RNN |

---

## Files

```
├── Weather_Forecasting_Astana.ipynb  ← Main Jupyter Notebook (all models)
├── app.py                            ← Streamlit web application
├── requirements.txt                  ← Python dependencies
├── astana_weather.csv                ← Generated after running notebook
└── models/                           ← Saved models (generated after notebook)
    ├── linear_regression.pkl
    ├── random_forest.pkl
    ├── xgboost.pkl
    ├── gru_model.keras
    ├── lstm_model.keras
    ├── scaler_X.pkl
    └── scaler_y.pkl
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Jupyter Notebook
Open `Weather_Forecasting_Astana.ipynb` in Jupyter and run all cells.  
This trains all 5 models and saves them to the `models/` folder.

### 3. Launch the Streamlit App
```bash
streamlit run app.py
```
Open your browser at: **http://localhost:8501**

---

## Dataset

Generated from real Astana climate statistics:
- **Period:** January 1, 2015 — December 31, 2024
- **Records:** 3,653 daily observations  
- **Features:** temp_mean, temp_max, temp_min, precipitation, windspeed, humidity, pressure

Climate characteristics of Astana:
- Strongly continental climate
- Winter: −20°C to −30°C
- Summer: +25°C to +35°C
- Annual mean: ~3.5°C

---

## Feature Engineering

19 input features per sample:
- **Raw:** 7 weather variables
- **Lag features:** t−1, t−2, t−3, t−7 for temperature and precipitation
- **Rolling stats:** 7-day and 14-day rolling mean/std
- **Cyclical encoding:** sin/cos of day-of-year for seasonality
- **Calendar:** month

---

## Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| MAE | Mean Absolute Error | Average error in °C |
| RMSE | Root Mean Squared Error | Penalizes large errors more |
| R² | Coefficient of Determination | 1.0 = perfect, 0 = mean baseline |

---

*Astana IT University | 2024–2025*
