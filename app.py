import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# -------------------------------
# Sidebar: Upload Data
# -------------------------------
st.sidebar.title("Upload Sales Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV with 'date' and 'sales' columns", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file, parse_dates=["date"], index_col="date")
    st.write("### Raw Data Preview")
    st.write(data.head())

    # Resample monthly
    monthly_sales = data["sales"].resample("M").sum()

    # -------------------------------
    # Train-Test Split
    # -------------------------------
    train_size = int(len(monthly_sales) * 0.8)
    train, test = monthly_sales[:train_size], monthly_sales[train_size:]

    st.write("### Monthly Sales Trend")
    st.line_chart(monthly_sales)

    # -------------------------------
    # ARIMA Forecast
    # -------------------------------
    st.subheader("ARIMA Forecast")
    model = ARIMA(train, order=(2,1,2))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test))

    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    st.write(f"**ARIMA MAE:** {mae:.2f}, **RMSE:** {rmse:.2f}")

    fig, ax = plt.subplots()
    ax.plot(train, label="Train")
    ax.plot(test, label="Test")
    ax.plot(test.index, forecast, label="Forecast")
    ax.legend()
    st.pyplot(fig)

    # -------------------------------
    # XGBoost Forecast
    # -------------------------------
    st.subheader("XGBoost Forecast")
    df = monthly_sales.reset_index()
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["lag1"] = df["sales"].shift(1)
    df["lag2"] = df["sales"].shift(2)
    df = df.dropna()

    train_size = int(len(df) * 0.8)
    train_df, test_df = df[:train_size], df[train_size:]

    X_train, y_train = train_df.drop(["sales","date"], axis=1), train_df["sales"]
    X_test, y_test = test_df.drop(["sales","date"], axis=1), test_df["sales"]

    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    st.write(f"**XGBoost MAE:** {mae:.2f}, **RMSE:** {rmse:.2f}")

    fig, ax = plt.subplots()
    ax.plot(y_test.values, label="Test")
    ax.plot(y_pred, label="Prediction")
    ax.legend()
    st.pyplot(fig)

    # -------------------------------
    # LSTM Forecast
    # -------------------------------
    st.subheader("LSTM Forecast")

    def create_sequences(series, window_size=12):
        X, y = [], []
        for i in range(len(series) - window_size):
            X.append(series[i:i+window_size])
            y.append(series[i+window_size])
        return np.array(X), np.array(y)

    series = monthly_sales.values
    X, y = create_sequences(series)

    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = Sequential([
        LSTM(50, activation="relu", input_shape=(X_train.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=20, verbose=0)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    st.write(f"**LSTM MAE:** {mae:.2f}, **RMSE:** {rmse:.2f}")

    fig, ax = plt.subplots()
    ax.plot(y_test, label="Test")
    ax.plot(y_pred, label="Prediction")
    ax.legend()
    st.pyplot(fig)
