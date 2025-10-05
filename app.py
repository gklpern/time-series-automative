import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go

# -------------------------------
# 1) Load trained models
# -------------------------------
prophet_bundle = joblib.load("models/otomotiv_satis/prophet_model.pkl")
prophet_model, bias = prophet_bundle["model"], prophet_bundle["bias"]

sarima_model = joblib.load("models/otomotiv_satis/sarima_model.pkl")

# -------------------------------
# 2) Load dataset (train + test with exogs)
# -------------------------------
train = pd.read_csv("data/final_data/train.csv")
test = pd.read_csv("data/final_data/test_filled.csv")

for df in [train, test]:
    df["Date"] = pd.to_datetime(df["Date"])

df_all = pd.concat([train, test], axis=0).set_index("Date").sort_index()
last_train_date = train["Date"].max()

# -------------------------------
# 3) User Inputs
# -------------------------------
st.title("Automotive Sales Forecast")

st.sidebar.header("Select Prediction Date")

date = st.sidebar.date_input(
    "Prediction Date",
    df_all.index.max(),  # default = son tarih
    min_value=df_all.index.min(),
    max_value=df_all.index.max()
)

# Tahmin yapılacak tarih = seçilen tarihten 1 ay gerisi
target_date = pd.to_datetime(date) - pd.DateOffset(months=1)

alpha = st.sidebar.slider("Prophet Weight (α)", 0.0, 1.0, 0.5, step=0.1)

# Crisis dummy
crisis_dummy = 1 if (pd.to_datetime("2018-01-01") <= target_date <= pd.to_datetime("2020-12-31")) else 0

# -------------------------------
# 4) Prepare Prophet input (same as notebook)
# -------------------------------
exog_vars = ["EUR/TL", "Faiz", "Kredi Stok", "OTV Orani"]

# Horizon (kaç ay ileri tahmin edileceğini bul)
delta_days = (target_date - last_train_date).days
steps_ahead = max(1, delta_days // 30 + 1)

# Prophet: train + steps_ahead kadar forecast
df_future = df_all[exog_vars].reset_index().rename(columns={"Date": "ds"})
df_future["crisis_dummy"] = ((df_future["ds"] >= "2018-01-01") & (df_future["ds"] <= "2020-12-31")).astype(int)

forecast_prophet = prophet_model.predict(df_future.iloc[:len(train)+steps_ahead])
yhat_prophet = forecast_prophet["yhat"].iloc[-1] + bias

# SARIMA: aynı horizon
sarima_forecast = sarima_model.get_forecast(steps=steps_ahead)
yhat_sarima = sarima_forecast.predicted_mean.iloc[-1]

# Ensemble
yhat_ensemble = alpha * yhat_prophet + (1 - alpha) * yhat_sarima

# -------------------------------
# 5) Output
# -------------------------------
st.markdown("### Forecast Result")
st.write(f"Selected Date: **{date.strftime('%Y-%m-%d')}**")
st.write(f"Forecast shown for: **{target_date.strftime('%Y-%m-%d')}** (1 month earlier)")

st.markdown(
    f"<h2 style='text-align: center; color: red;'>Ensemble Forecast: {int(yhat_ensemble):,}</h2>", 
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)
col1.metric("Prophet+Exog Forecast", f"{int(yhat_prophet):,}")
col2.metric("SARIMA Forecast", f"{int(yhat_sarima):,}")

st.caption(f"Ensemble = α·Prophet + (1-α)·SARIMA   |   Current α = {alpha:.1f}")

with st.expander("Exogenous Variables on Selected Date"):
    if target_date in df_all.index:
        exog_values = df_all.loc[target_date, exog_vars].to_dict()
        st.table(pd.DataFrame([exog_values], index=[target_date]))
    else:
        st.write("Selected target date not in dataset (future extrapolation).")

with st.expander("Debug Information"):
    st.write(f"**SARIMA Steps Ahead:** {steps_ahead}")
    st.write(f"**Last Train Date:** {last_train_date.strftime('%Y-%m-%d')}")
    st.write(f"**Target Prediction Date:** {target_date.strftime('%Y-%m-%d')}")
    st.write(f"**Days Difference:** {delta_days}")

# -------------------------------
# 6) Plot
# -------------------------------
fig = go.Figure()

# Train series
fig.add_trace(go.Scatter(
    x=df_all.index, y=df_all["Otomotiv Satis"],
    mode="lines",
    name="Train (Actual)",
    line=dict(color="blue")
))

# Prophet prediction point
fig.add_trace(go.Scatter(
    x=[target_date], y=[yhat_prophet],
    mode="markers",
    name="Prophet+Exog Forecast",
    marker=dict(color="green", size=12, symbol="circle")
))

# SARIMA prediction point
fig.add_trace(go.Scatter(
    x=[target_date], y=[yhat_sarima],
    mode="markers",
    name="SARIMA Forecast",
    marker=dict(color="magenta", size=12, symbol="diamond")
))

# Ensemble prediction point
fig.add_trace(go.Scatter(
    x=[target_date], y=[yhat_ensemble],
    mode="markers+text",
    name="Ensemble Forecast",
    marker=dict(color="red", size=14, symbol="star"),
    text=[f"{int(yhat_ensemble):,}"],
    textposition="top center"
))

# Vertical line at target date
fig.add_vline(x=target_date, line=dict(color="gray", dash="dash"))

fig.update_layout(
    title="Automotive Sales Forecast (Prediction for 1 Month Earlier)",
    xaxis_title="Date",
    yaxis_title="Sales",
    template="plotly_white",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)
