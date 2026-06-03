# Business Sales Forecasting App

## Overview
This Streamlit app forecasts sales using:
- ARIMA (time-series)
- XGBoost (machine learning)
- LSTM (deep learning)

It supports both **CSV and Excel uploads** with `date` and `sales` columns.

## Features
- Upload your sales dataset
- Interactive charts
- Model evaluation (MAE, RMSE)
- Forecast visualization

## Requirements
See `requirements.txt`.

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
