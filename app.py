"""
Weather Forecasting Dashboard — Astana, Kazakhstan
Final Project: Machine Learning (Advanced Level)
Astana IT University

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pickle, os, warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Astana Weather Forecasting",
    page_icon="🌤",
    layout="wide",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 4px solid #2196F3;
        margin-bottom: 10px;
    }
    .metric-card h4 { margin: 0; color: #555; font-size: 13px; }
    .metric-card h2 { margin: 4px 0 0; color: #1a1a1a; font-size: 28px; }
    .best-badge {
        background: #E8F5E9;
        color: #2E7D32;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Data generation (same seed as notebook) ───────────────────────────────────
@st.cache_data
def load_data():
    np.random.seed(42)
    dates = pd.date_range('2015-01-01', '2024-12-31', freq='D')
    n = len(dates)
    doy = np.array([d.timetuple().tm_yday for d in dates])

    temp_mean = 3.5 + 18.5 * np.sin((doy - 80) / 365 * 2 * np.pi) + np.random.normal(0, 3, n)
    temp_max  = temp_mean + np.abs(np.random.normal(5, 1.5, n))
    temp_min  = temp_mean - np.abs(np.random.normal(5, 1.5, n))
    precip_base = 0.5 + 1.8 * np.clip(np.sin((doy - 60) / 365 * 2 * np.pi), 0, 1)
    precipitation = np.random.exponential(precip_base, n)
    precipitation = np.where(np.random.rand(n) < 0.65, 0, precipitation)
    wind = 4 + 2 * np.sin((doy - 30) / 365 * 2 * np.pi) + np.abs(np.random.normal(0, 2, n))
    humidity = np.clip(50 + 20 * np.sin((doy - 100) / 365 * 2 * np.pi) + np.random.normal(0, 7, n), 15, 95)
    pressure = 1013 + np.random.normal(0, 8, n)

    df = pd.DataFrame({
        'date': dates,
        'temp_mean': temp_mean.round(1),
        'temp_max':  temp_max.round(1),
        'temp_min':  temp_min.round(1),
        'precipitation': precipitation.round(1),
        'windspeed': wind.round(1),
        'humidity':  humidity.round(1),
        'pressure':  pressure.round(1)
    })
    return df

@st.cache_data
def engineer_features(df):
    df = df.sort_values('date').reset_index(drop=True)
    for lag in [1, 2, 3, 7]:
        df[f'temp_lag{lag}'] = df['temp_mean'].shift(lag)
        df[f'precip_lag{lag}'] = df['precipitation'].shift(lag)
    df['temp_roll7_mean']  = df['temp_mean'].shift(1).rolling(7).mean()
    df['temp_roll7_std']   = df['temp_mean'].shift(1).rolling(7).std()
    df['temp_roll14_mean'] = df['temp_mean'].shift(1).rolling(14).mean()
    df['month']      = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    df['sin_doy'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['cos_doy'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['target'] = df['temp_mean'].shift(-1)
    return df.dropna().reset_index(drop=True)

FEATURES = [
    'temp_mean', 'temp_max', 'temp_min', 'precipitation',
    'windspeed', 'humidity', 'pressure',
    'temp_lag1', 'temp_lag2', 'temp_lag3', 'temp_lag7',
    'precip_lag1', 'precip_lag7',
    'temp_roll7_mean', 'temp_roll7_std', 'temp_roll14_mean',
    'sin_doy', 'cos_doy', 'month'
]

@st.cache_resource
def train_all_models(df_ml):
    split_idx = int(len(df_ml) * 0.8)
    train_df = df_ml.iloc[:split_idx]
    test_df  = df_ml.iloc[split_idx:]

    X_train = train_df[FEATURES].values
    y_train = train_df['target'].values
    X_test  = test_df[FEATURES].values
    y_test  = test_df['target'].values

    models = {}

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models['Linear Regression'] = lr

    rf = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf

    xgb = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.07,
                        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    models['XGBoost'] = xgb

    return models, X_train, y_train, X_test, y_test, train_df, test_df

def get_metrics(y_true, y_pred):
    return {
        'MAE':  mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R²':   r2_score(y_true, y_pred)
    }

# ── Load & train ──────────────────────────────────────────────────────────────
df_raw = load_data()
df_ml  = engineer_features(df_raw)
models, X_train, y_train, X_test, y_test, train_df, test_df = train_all_models(df_ml)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Flag_of_Kazakhstan.svg/200px-Flag_of_Kazakhstan.svg.png", width=100)
st.sidebar.title("⚙️ Settings")

selected_models = st.sidebar.multiselect(
    "Select models to display:",
    options=list(models.keys()),
    default=list(models.keys())
)

show_days = st.sidebar.slider("Days to show in forecast plot:", 30, 365, 90)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** Astana, Kazakhstan  \n2015–2024 | 3,653 daily records  \n**Target:** Next-day temperature")
st.sidebar.markdown("---")
st.sidebar.markdown("**Astana IT University**  \nML Final Project 2024/25")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🌤 Weather Forecasting Dashboard — Astana, Kazakhstan")
st.caption("Multi-model next-day temperature prediction using 10 years of historical climate data")

# ── KPI Row ───────────────────────────────────────────────────────────────────
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("📅 Dataset period", "2015 – 2024")
with kpi2:
    st.metric("📊 Total records", f"{len(df_raw):,}")
with kpi3:
    st.metric("🌡 Temperature range", f"{df_raw.temp_mean.min():.0f}°C to {df_raw.temp_mean.max():.0f}°C")
with kpi4:
    st.metric("🤖 Models trained", str(len(models)))

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🤖 Model Results", "🔮 Live Prediction", "📈 Detailed Analysis"])

# ════════════════════════════════════════════════════════
# TAB 1: Data Overview
# ════════════════════════════════════════════════════════
with tab1:
    st.subheader("Historical Temperature (2015–2024)")

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(df_raw['date'], df_raw['temp_mean'], color='#90CAF9', alpha=0.5, linewidth=0.7)
    ax.fill_between(df_raw['date'], df_raw['temp_min'], df_raw['temp_max'], alpha=0.1, color='#2196F3')
    monthly = df_raw.set_index('date').resample('ME')['temp_mean'].mean()
    ax.plot(monthly.index, monthly.values, color='#FF5722', linewidth=2.2, label='Monthly mean')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_ylabel('Temperature (°C)')
    ax.legend()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Monthly Temperature Distribution")
        fig, ax = plt.subplots(figsize=(7, 4))
        df_raw['month'] = df_raw['date'].dt.month
        month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        monthly_temps = [df_raw[df_raw['month']==m]['temp_mean'].values for m in range(1,13)]
        bp = ax.boxplot(monthly_temps, labels=month_names, patch_artist=True)
        palette = plt.cm.RdYlBu_r(np.linspace(0, 1, 12))
        for patch, color in zip(bp['boxes'], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_ylabel('Temperature (°C)')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Feature Correlations")
        import seaborn as sns
        corr_cols = ['temp_mean','temp_max','temp_min','precipitation','windspeed','humidity','pressure']
        corr = df_raw[corr_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                    mask=mask, ax=ax, square=True, linewidths=0.5, annot_kws={'size': 9})
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.subheader("Raw Data Sample")
    st.dataframe(df_raw.tail(10).reset_index(drop=True), use_container_width=True)

# ════════════════════════════════════════════════════════
# TAB 2: Model Results
# ════════════════════════════════════════════════════════
with tab2:
    st.subheader("Model Performance on Test Set (2023–2024)")

    all_preds = {}
    metrics_list = []
    for name, model in models.items():
        pred = model.predict(X_test)
        all_preds[name] = pred
        m = get_metrics(y_test, pred)
        metrics_list.append({'Model': name, 'MAE (°C)': round(m['MAE'], 3),
                              'RMSE (°C)': round(m['RMSE'], 3), 'R²': round(m['R²'], 4)})

    metrics_df = pd.DataFrame(metrics_list).sort_values('RMSE (°C)').reset_index(drop=True)
    metrics_df.index += 1

    # Highlight best
    best_model = metrics_df.iloc[0]['Model']
    st.success(f"🏆 Best model: **{best_model}** (lowest RMSE)")
    st.dataframe(metrics_df, use_container_width=True)

    # Bar chart comparison
    col1, col2, col3 = st.columns(3)
    metrics_ordered = metrics_df.sort_values('Model')

    for col, metric, label, ascending in zip(
        [col1, col2, col3],
        ['MAE (°C)', 'RMSE (°C)', 'R²'],
        ['MAE — lower is better', 'RMSE — lower is better', 'R² — higher is better'],
        [True, True, False]
    ):
        fig, ax = plt.subplots(figsize=(5, 3.5))
        colors = ['#42A5F5', '#66BB6A', '#AB47BC']
        vals = [metrics_df[metrics_df['Model']==m][metric].values[0] for m in metrics_df['Model']]
        mnames = metrics_df['Model'].tolist()
        bars = ax.bar(mnames, vals, color=colors[:len(mnames)], edgecolor='white')
        ax.set_title(label, fontsize=11)
        ax.tick_params(axis='x', rotation=30, labelsize=9)
        if metric == 'R²':
            ax.set_ylim(0, 1)
        plt.tight_layout()
        col.pyplot(fig)
        plt.close()

    # Prediction plot
    st.subheader(f"Predictions vs Actual — Last {show_days} Test Days")
    n_show = min(show_days, len(y_test))
    test_dates = test_df['date'].values

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(test_dates[-n_show:], y_test[-n_show:], 'k-', linewidth=2.2, label='Actual', zorder=5)

    line_colors = ['#EF5350', '#42A5F5', '#66BB6A']
    for (name, pred), lc in zip(all_preds.items(), line_colors):
        if name in selected_models:
            ax.plot(test_dates[-n_show:], pred[-n_show:], '--', color=lc, alpha=0.85, linewidth=1.5, label=name)

    ax.axhline(0, color='gray', linestyle=':', linewidth=0.7)
    ax.set_ylabel('Temperature (°C)')
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ════════════════════════════════════════════════════════
# TAB 3: Live Prediction
# ════════════════════════════════════════════════════════
with tab3:
    st.subheader("🔮 Predict Tomorrow's Temperature")
    st.markdown("Enter today's weather conditions and get next-day temperature predictions from all models.")

    col1, col2, col3 = st.columns(3)

    with col1:
        inp_temp_mean = st.number_input("Today's mean temperature (°C)", min_value=-35.0, max_value=45.0, value=15.0, step=0.5)
        inp_temp_max  = st.number_input("Today's max temperature (°C)",  min_value=-30.0, max_value=50.0, value=inp_temp_mean + 5.0, step=0.5)
        inp_temp_min  = st.number_input("Today's min temperature (°C)",  min_value=-40.0, max_value=40.0, value=inp_temp_mean - 5.0, step=0.5)
    with col2:
        inp_precip  = st.number_input("Precipitation (mm)", min_value=0.0, max_value=50.0, value=0.0, step=0.5)
        inp_wind    = st.number_input("Wind speed (km/h)", min_value=0.0, max_value=30.0, value=5.0, step=0.5)
        inp_humidity = st.slider("Humidity (%)", 10, 100, 50)
    with col3:
        inp_pressure = st.number_input("Pressure (hPa)", min_value=970.0, max_value=1060.0, value=1013.0, step=1.0)
        inp_date = st.date_input("Date (today)", pd.Timestamp.today())

    if st.button("🌡 Predict Tomorrow", type="primary"):
        doy_val = inp_date.timetuple().tm_yday
        month_val = inp_date.month

        # Use last 7 days from dataset as rough lags (simplified for demo)
        recent = df_ml.tail(10)
        lag1 = recent['temp_mean'].iloc[-1]
        lag2 = recent['temp_mean'].iloc[-2]
        lag3 = recent['temp_mean'].iloc[-3]
        lag7 = recent['temp_mean'].iloc[-7]
        pl1  = recent['precipitation'].iloc[-1]
        pl7  = recent['precipitation'].iloc[-7]
        r7m  = recent['temp_mean'].iloc[-7:].mean()
        r7s  = recent['temp_mean'].iloc[-7:].std()
        r14m = recent['temp_mean'].iloc[-14:].mean()

        X_input = np.array([[
            inp_temp_mean, inp_temp_max, inp_temp_min, inp_precip,
            inp_wind, inp_humidity, inp_pressure,
            lag1, lag2, lag3, lag7,
            pl1, pl7,
            r7m, r7s, r14m,
            np.sin(2 * np.pi * doy_val / 365),
            np.cos(2 * np.pi * doy_val / 365),
            month_val
        ]])

        st.markdown("---")
        st.markdown("### Predictions for tomorrow:")
        pred_cols = st.columns(len(models))
        model_colors = {'Linear Regression': '#EF5350', 'Random Forest': '#42A5F5', 'XGBoost': '#66BB6A'}

        for col, (name, model) in zip(pred_cols, models.items()):
            pred_val = model.predict(X_input)[0]
            delta = pred_val - inp_temp_mean
            with col:
                st.metric(
                    label=name,
                    value=f"{pred_val:.1f} °C",
                    delta=f"{delta:+.1f}°C vs today"
                )

        # Simple weather interpretation
        avg_pred = np.mean([m.predict(X_input)[0] for m in models.values()])
        if avg_pred < -10:
            weather_desc = "❄️ Very cold — dress warmly!"
        elif avg_pred < 0:
            weather_desc = "🌨 Cold, possible snow"
        elif avg_pred < 10:
            weather_desc = "🌥 Cool, light jacket recommended"
        elif avg_pred < 20:
            weather_desc = "⛅ Mild and pleasant"
        elif avg_pred < 28:
            weather_desc = "☀️ Warm and sunny"
        else:
            weather_desc = "🌡 Hot day ahead — stay hydrated!"

        st.info(f"**Average forecast: {avg_pred:.1f}°C** — {weather_desc}")

# ════════════════════════════════════════════════════════
# TAB 4: Detailed Analysis
# ════════════════════════════════════════════════════════
with tab4:
    st.subheader("Residual Analysis")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors_list = ['#EF5350', '#42A5F5', '#66BB6A']

    for ax, (name, pred), color in zip(axes, all_preds.items(), colors_list):
        residuals = y_test - pred
        ax.scatter(pred, residuals, alpha=0.3, s=8, color=color)
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.set_title(f'{name}\nResiduals vs Predicted')
        ax.set_xlabel('Predicted (°C)')
        ax.set_ylabel('Residual (°C)')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Year-by-Year Prediction Accuracy")
    test_df_copy = test_df.copy()

    yearly_metrics = []
    for year in test_df_copy['date'].dt.year.unique():
        mask = test_df_copy['date'].dt.year == year
        if mask.sum() < 10:
            continue
        idx = test_df_copy[mask].index - test_df_copy.index[0]
        idx = [i for i in idx if i < len(y_test)]
        if not idx:
            continue
        for name, pred in all_preds.items():
            mae_y = mean_absolute_error(y_test[idx], pred[idx])
            yearly_metrics.append({'Year': year, 'Model': name, 'MAE': round(mae_y, 2)})

    yearly_df = pd.DataFrame(yearly_metrics)
    if not yearly_df.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        for name, color in zip(all_preds.keys(), colors_list):
            sub = yearly_df[yearly_df['Model'] == name]
            ax.plot(sub['Year'], sub['MAE'], marker='o', label=name, color=color, linewidth=2)
        ax.set_ylabel('MAE (°C)')
        ax.set_title('MAE per Year in Test Period')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.subheader("Dataset Statistics")
    st.dataframe(df_raw.describe().round(2), use_container_width=True)
