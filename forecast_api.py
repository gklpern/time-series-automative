from fastapi import FastAPI
import pandas as pd
import joblib
from pydantic import BaseModel

# -------------------------------
# 1) Load models
# -------------------------------
prophet_bundle = joblib.load("models/otomotiv_satis/prophet_model.pkl")
prophet_model, bias = prophet_bundle["model"], prophet_bundle["bias"]
sarima_model = joblib.load("models/otomotiv_satis/sarima_model.pkl")

# -------------------------------
# 2) Load data (train + test)
# -------------------------------
train = pd.read_csv("data/final_data/train.csv")
test = pd.read_csv("data/final_data/test_filled.csv")

for df in [train, test]:
    df["Date"] = pd.to_datetime(df["Date"])

train = train.set_index("Date")
test = test.set_index("Date")

df_all = pd.concat([train, test]).sort_index()

# -------------------------------
# 3) FastAPI app
# -------------------------------
app = FastAPI(title="Automotive Sales Forecast API")

# Request schema for /forecast
class ForecastRequest(BaseModel):
    date: str
    EUR_TL: float
    Faiz: float
    Kredi_Stok: float
    OTV_Orani: float
    alpha: float = 0.5


# -------------------------------
# 4) Endpoint: Get exog values (1 month earlier)
# -------------------------------
@app.get("/get_inputs")
def get_inputs(date: str, alpha: float = 0.5):
    # Kullanıcının seçtiği tarih
    user_date = pd.to_datetime(date)
    
    # Tahmin yapılacak tarih = seçilen tarihten 1 ay gerisi (app.py ile aynı mantık)
    target_date = user_date - pd.DateOffset(months=1)

    if target_date not in df_all.index:
        return {
            "error": f"{target_date.strftime('%Y-%m-%d')} not in dataset "
                     f"(available: {df_all.index.min()} → {df_all.index.max()})"
        }

    row = df_all.loc[target_date, ["EUR/TL", "Faiz", "Kredi Stok", "OTV Orani"]]

    return {
        "date": user_date.strftime("%Y-%m-%d"),   # kullanıcının girdiği tarih
        "EUR_TL": float(row["EUR/TL"]),
        "Faiz": float(row["Faiz"]),
        "Kredi_Stok": float(row["Kredi Stok"]),
        "OTV_Orani": float(row["OTV Orani"]),
        "alpha": alpha
    }


# -------------------------------
# 5) Endpoint: Forecast
# -------------------------------
@app.post("/forecast")
def forecast(req: ForecastRequest):
    # Kullanıcının seçtiği tarih
    user_date = pd.to_datetime(req.date)
    
    # Tahmin yapılacak tarih = seçilen tarihten 1 ay gerisi (app.py ile aynı mantık)
    target_date = user_date - pd.DateOffset(months=1)

    # Crisis dummy (target_date için)
    crisis_dummy = 1 if (pd.to_datetime("2018-01-01") <= target_date <= pd.to_datetime("2020-12-31")) else 0

    # Exog değerlerini target_date'den al
    if target_date not in df_all.index:
        return {"error": f"{target_date.strftime('%Y-%m-%d')} not in dataset"}

    exog_row = df_all.loc[target_date, ["EUR/TL", "Faiz", "Kredi Stok", "OTV Orani"]]

    # Prophet input (app.py ile aynı mantık)
    exog_vars = ["EUR/TL", "Faiz", "Kredi Stok", "OTV Orani"]
    
    # Horizon hesaplama (app.py ile aynı)
    last_train_date = train.index.max()
    delta_days = (target_date - last_train_date).days
    steps_ahead = max(1, delta_days // 30 + 1)

    # Prophet: train + steps_ahead kadar forecast (app.py ile aynı)
    df_future = df_all[exog_vars].reset_index().rename(columns={"Date": "ds"})
    df_future["crisis_dummy"] = ((df_future["ds"] >= "2018-01-01") & (df_future["ds"] <= "2020-12-31")).astype(int)

    forecast_prophet = prophet_model.predict(df_future.iloc[:len(train)+steps_ahead])
    yhat_prophet = forecast_prophet["yhat"].iloc[-1] + bias

    # SARIMA: aynı horizon (app.py ile aynı)
    sarima_forecast = sarima_model.get_forecast(steps=steps_ahead)
    yhat_sarima = sarima_forecast.predicted_mean.iloc[-1]

    # Ensemble
    yhat_ensemble = req.alpha * yhat_prophet + (1 - req.alpha) * yhat_sarima

    return {
        "date": req.date,  # kullanıcının seçtiği tarih
        "prophet_forecast": float(yhat_prophet),
        "sarima_forecast": float(yhat_sarima),
        "ensemble_forecast": float(yhat_ensemble),
        "alpha": req.alpha
    }


# -------------------------------
# 6) Endpoint: Simple Ensemble Forecast
# -------------------------------
@app.post("/predict")
def predict(req: ForecastRequest):
    # Kullanıcının seçtiği tarih
    user_date = pd.to_datetime(req.date)
    
    # Tahmin yapılacak tarih = seçilen tarihten 1 ay gerisi
    target_date = user_date - pd.DateOffset(months=1)

    # Crisis dummy (target_date için)
    crisis_dummy = 1 if (pd.to_datetime("2018-01-01") <= target_date <= pd.to_datetime("2020-12-31")) else 0

    # Exog değerlerini target_date'den al
    if target_date not in df_all.index:
        return {"error": f"{target_date.strftime('%Y-%m-%d')} not in dataset"}

    exog_row = df_all.loc[target_date, ["EUR/TL", "Faiz", "Kredi Stok", "OTV Orani"]]

    # Prophet input
    exog_vars = ["EUR/TL", "Faiz", "Kredi Stok", "OTV Orani"]
    
    # Horizon hesaplama
    last_train_date = train.index.max()
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
    yhat_ensemble = req.alpha * yhat_prophet + (1 - req.alpha) * yhat_sarima

    return {
        "date": req.date,
        "prediction": float(yhat_ensemble)
    }
