import streamlit as st
import holidays
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.volatility import BollingerBands
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

# Set page config
st.set_page_config(page_title="Stock App", layout="wide")

# Title
st.title("📈 Stock Data Fetcher with TA Indicators")

# Sidebar Inputs
st.sidebar.header("User Input Parameters")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., TCS.NS)", value="TCS.NS")
period_options = ['3mo', '6mo', '1y', '2y', '5y', 'max']
period = st.sidebar.selectbox("Select Period", options=period_options, index=2)

# Fetch stock data with NSE holidays and weekends removed
def fetch_stock_data(ticker, period='6mo'):
    interval_mapping = {
        '3mo': '1d',
        '6mo': '1d',
        '1y': '1d',
        '2y': '1d',
        '5y': '1d',
        'max': '1mo'
    }
    interval = interval_mapping.get(period, '1d')

    try:
        df = yf.download(ticker, period=period, interval=interval, end=datetime.today() + timedelta(days=1))

        if df.empty:
            st.warning("No data fetched. Please try a different period.")
            return None

        # Ensure datetime index is timezone-aware (UTC) before converting
        df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('Asia/Kolkata')

        # Remove weekends and NSE holidays
        df = df[df.index.dayofweek < 5]  # Remove weekends
        nse_holidays = holidays.India(years=df.index[-1].year)
        df = df[~df.index.to_series().dt.date.isin(nse_holidays)]

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df.columns.values]
        df.columns = [col.split('_')[0] if '_' in col else col for col in df.columns]

    df.sort_index(inplace=True)

    if df.empty or len(df) < 30:
        st.warning("Not enough valid trading days. Try a longer period like 3mo or 6mo.")

    return df

# Apply indicators based on chosen tier
def add_technical_indicators(df):
    df = df.copy()

    df['EMA_12'] = EMAIndicator(close=df['Close'], window=12).ema_indicator()
    df['EMA_26'] = EMAIndicator(close=df['Close'], window=26).ema_indicator()

    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()

    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_lower'] = bb.bollinger_lband()
    df['BB_middle'] = bb.bollinger_mavg()

    return df.dropna()

def prepare_features_for_regression(df):
    df = df.copy()

    # LAG FEATURES (last 3 days)
    for lag in range(1, 4):
        df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
        df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)

    # ROLLING MEAN & STD
    df['RollingMean_3'] = df['Close'].rolling(window=3).mean()
    df['RollingStd_3'] = df['Close'].rolling(window=3).std()
    df['RollingMean_7'] = df['Close'].rolling(window=7).mean()
    df['RollingStd_7'] = df['Close'].rolling(window=7).std()

    # DAILY RETURNS
    df['Return_1d'] = df['Close'].pct_change()
    df['LogReturn_1d'] = (df['Close'] / df['Close'].shift(1)).apply(lambda x: np.log(x) if x > 0 else 0)

    # TARGETS (next 1 to 3 day closing prices)
    for horizon in range(1, 4):
        df[f'Target_Close_{horizon}d'] = df['Close'].shift(-horizon)

    return df

nse_holidays = holidays.India(years=range(2020, 2035))  # Wider range if needed

def get_next_trading_day(date):
    next_day = date + pd.Timedelta(days=1)
    while next_day.weekday() >= 5 or next_day.date() in nse_holidays:
        next_day += pd.Timedelta(days=1)
    return next_day

def train_xgboost_model(df, n_days=3):
    df = df.copy()

    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
                    'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal', 
                    'LogReturn_1d', 'BB_upper', 'BB_middle', 'BB_lower']
    
    df.dropna(subset=feature_cols, inplace=True)
    df['Target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)

    X = df[feature_cols]
    y = df['Target']

    # Time series split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=False, test_size=0.1
    )

    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Standard test prediction
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    accuracy = 100 - (mape * 100)

    print("Model Evaluation:")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")

    # === Recursive Forecasting for Next n_days ===
    last_row = df.iloc[-1].copy()
    last_date = df.index[-1]  # latest trading date in dataset

    # Start forecast from the next trading day AFTER today
    today = pd.Timestamp.now(tz='Asia/Kolkata').normalize()
    forecast_start = get_next_trading_day(today)

    future_dates = []
    predictions = []

    while len(predictions) < n_days:
        input_features = last_row[feature_cols].copy()
        pred_close = model.predict(input_features.values.reshape(1, -1))[0]

        predictions.append(pred_close)
        future_dates.append(forecast_start)

        # Update last_row for next prediction step
        last_row['Close'] = pred_close
        last_row['LogReturn_1d'] = np.log(pred_close / input_features['Close']) if input_features['Close'] != 0 else 0

        forecast_start = get_next_trading_day(forecast_start)

    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Close': predictions
    }).set_index('Date')

    return model, X_test, y_test, y_pred, forecast_df

#streamlit
if st.sidebar.button("Fetch Data"):
    df = fetch_stock_data(ticker, period)

    if df is not None and not df.empty:
        st.success(f"Data fetched for {ticker} - {period}")
        df = add_technical_indicators(df)
        df = prepare_features_for_regression(df) 

        # Display final processed data
        df1 = df.fillna(0)
        st.subheader("Filled Preview")
        st.dataframe(df1.tail(15))

        df.dropna(inplace=True)
        st.subheader("📊 XGBoost Model Training & Forecast")
        model, X_test, y_test, y_pred, forecast_df = train_xgboost_model(df)

        st.write("Model trained on past data. Here's how it performed on recent test data:")

        # Use get_next_trading_day to get actual next trading day for each test sample
        test_dates = pd.Series(X_test.index).apply(get_next_trading_day)

        results_df = pd.DataFrame({
                'Actual': y_test.values,
                'Predicted': y_pred
            }, index=test_dates)
    
        st.dataframe(results_df.tail(10))

        st.subheader("🔮 Next 3-Day Forecast")
        st.dataframe(forecast_df)

    else:
        st.warning("No data found for this ticker or period.")