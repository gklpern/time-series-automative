# Automotive Sales Forecast  

Bu proje, Türkiye’de otomotiv satışlarını tahmin etmeye yönelik Prophet ve SARIMA tabanlı zaman serisi modellerini bir araya getiren hibrit bir yaklaşımdır. Prophet modeli ekonomik göstergeler (EUR/TL, faiz oranı, kredi stoku, ÖTV oranı) gibi dışsal değişkenlerle eğitilmiş, SARIMA modeli ise yalnızca serinin kendi dinamiklerini kullanmıştır. İki model, Ensemble yöntemiyle birleştirilmiştir.  

## İçerik  
- **main_notebook.ipynb**: Veri ön işleme, model geliştirme, çapraz doğrulama, metriklerin hesaplanması ve tahmin görselleştirmeleri.  
- **forecast_api.py**: FastAPI tabanlı servis. JSON formatında parametre alır, Prophet, SARIMA ve Ensemble tahminlerini döndürür.  
- **app.py**: Streamlit tabanlı kullanıcı arayüzü (demo amaçlı).  
- **models/**: Kaydedilmiş Prophet ve SARIMA modelleri.  
- **Dockerfile**: Servisin Docker imajı olarak paketlenmesini sağlar.  
- **requirements.txt**: Bağımlılık listesi.  
- **presentation/**: Projeye ait sunum dosyası.  

## Kullanım  

### Model Geliştirme  
Notebook dosyasında veri yüklenir, exog değişkenler hazırlanır ve modeller eğitilir. Sonrasında modeller `models/otomotiv_satis/` klasörüne kaydedilir.  

# API Servisi  
API, FastAPI ile geliştirilmiştir. Endpointler:  


## GET/get_inputs

get_inputs ile kullanıcı, talep edilen tahmin tarihi için kullanılacak bağımsız değişkenleri (EUR/TL, ÖTV vs.) alabilir.

GET http://localhost:8000/get_inputs?date=2022-07-01

Çıktı:



```json
{
    "date": "2022-07-01",
    "EUR_TL": 16.58113499667224,
    "Faiz": 28.072871151121323,
    "Kredi_Stok": 5060853.234431182,
    "OTV_Orani": 65.0,
    "alpha": 0.5
}

## POST/forecast

forecast ile tarih ve bağımsız değişkenlere bağlı olarak 3 farklı model için tahmin çıktısı alınır. Ensemble Çıktısı, 2 modelin Hybrid çalışmasının çıktısını verir. Parametre ayarlaması yapılabilir.

POST http://localhost:8000/forecast

Girdi: get_inputs çıktısı bu endpointe girdi olarak verilebilir. (JSON Formatında)

{
    "date": "2022-07-01",
    "EUR_TL": 16.58113499667224,
    "Faiz": 28.072871151121323,
    "Kredi_Stok": 5060853.234431182,
    "OTV_Orani": 65.0,
    "alpha": 0.5
}

Çıktı:

{
    "date": "2022-07-01",
    "prophet_forecast": 46370.25563266318,
    "sarima_forecast": 43316.95249260511,
    "ensemble_forecast": 44843.60406263414,
    "alpha": 0.5
}


## POST/predict

predict ile sadece final modelimiz olan hybrid modelimizin çıktısı ve tarih çıktı olarak verilir.

Girdi:
{
    "date": "2022-07-01",
    "EUR_TL": 16.58113499667224,
    "Faiz": 28.072871151121323,
    "Kredi_Stok": 5060853.234431182,
    "OTV_Orani": 65.0,
    "alpha": 0.5
}

Çıktı:

{
    "date": "2022-07-01",
    "prediction": 44843.60406263414
}





#Docker ile Çalıştırma

Proje klasöründeyken:

docker build -t automotive-forecast .
docker run -p 8000:8000 automotive-forecast




Docker run edildikten sonra "http://localhost:8000/docs" adresinde çalışır.


"http://localhost:8000/forecast" adresine POST atabilirsiniz.