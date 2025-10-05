FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY forecast_api.py .
COPY models/ ./models/
COPY data/ data/

EXPOSE 8000

CMD ["uvicorn", "forecast_api:app", "--host", "0.0.0.0", "--port", "8000"]
