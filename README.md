# AirAware AI — Air Quality Forecaster & Health Advisor

A Streamlit portfolio project forecasting PM2.5 for 24 or 48 hours with a Random Forest regressor and giving cautious, plain-language health tips.

## Run locally

Python 3.10–3.12 recommended.

### Windows PowerShell
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL Streamlit prints. Live mode needs internet access. Turn on **Use included demo dataset** to explore offline using synthetic illustrative data.

## Optional LLM
Copy `.env.example` to `.env` and add your `OPENAI_API_KEY`. The app loads `.env` automatically. Keep secrets private. Without a key, built-in rules generate the advice.

## Data sources
- Open-Meteo Air Quality API: https://open-meteo.com/en/docs/air-quality-api
- Open-Meteo Weather API: https://open-meteo.com/en/docs

Check current API terms and usage limits before commercial deployment.

## Model details
- Target: PM2.5 concentration in µg/m³.
- Features: 1, 2, 3, 6, 12, 24, 48-hour lags; rolling means; hour/day-of-week; temperature, humidity, wind.
- Evaluation: chronological holdout of final 20%, with MAE/RMSE compared to previous-hour persistence baseline.
- Forecast: recursive one-hour-ahead prediction; errors may accumulate.
- AQI: approximate US EPA-style PM2.5-only conversion, not the Indian CPCB scale or a multi-pollutant AQI.
- Health advice is educational, not medical advice. Follow local official alerts.

## Interview summary
“I built an end-to-end ML app that collects hourly environmental data, engineers time-series features, evaluates a Random Forest against a persistence baseline, recursively forecasts PM2.5 using weather features, and translates predicted pollution into cautious guidance. It also includes offline demo data and CSV exports.”
