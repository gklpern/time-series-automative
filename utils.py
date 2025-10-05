import pandas as pd
import numpy as np

def calculate_steps_ahead(selected_date, last_train_date):
    """
    Kullanıcının seçtiği tarih -> target_date = 1 ay öncesi.
    Streamlit'in 'doğru çalışan' versiyonuna göre steps hesaplanır.
    """
    target_date = pd.to_datetime(selected_date) - pd.DateOffset(months=1)
    delta_days = (target_date - pd.to_datetime(last_train_date)).days
    steps_ahead = max(1, delta_days // 30 + 1)
    return steps_ahead, delta_days, target_date


def apply_bias_correction(model, df_train, y_train):
    """
    Prophet için bias correction hesaplar.
    Eğitim setindeki gerçek değerler - tahmin farklarının ortalamasını döndürür.
    """
    yhat_train = model.predict(df_train)["yhat"].values
    bias = np.mean(y_train.values - yhat_train)
    return bias
