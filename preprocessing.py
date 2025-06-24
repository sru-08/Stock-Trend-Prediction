import streamlit as st
import holidays
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.volatility import BollingerBands
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

# Set page config
st.set_page_config(page_title="Stock App", layout="wide")

# Title
st.title("📈 Stock Market Dashboard")

# Sidebar Inputs
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., TCS.NS)", value="TCS.NS")

# Calculate default dates - ensuring enough data for model training
# Model needs at least 30 days + lag features + technical indicators
# Setting default to ~6 months (180 days) to ensure sufficient trading days
default_end_date = datetime.today().date()
default_start_date = (datetime.today() - timedelta(days=180)).date()

start_date = st.sidebar.date_input("Start Date", value=default_start_date)
end_date = st.sidebar.date_input("End Date", value=default_end_date)

# Input validation
date_validation_passed = False
if start_date and end_date:
    if start_date >= end_date:
        st.sidebar.error("⚠️ Start date must be before end date")
    elif (end_date - start_date).days < 180:
        days_diff = (end_date - start_date).days
        st.sidebar.error(f"⚠️ Date range is {days_diff} days. Need at least 180 days for proper model training")
    else:
        days_diff = (end_date - start_date).days
        st.sidebar.success(f"✓ Date range: {days_diff} days (sufficient for training)")
        date_validation_passed = True

# Fetch stock data with NSE holidays and weekends removed
def fetch_stock_data(ticker, start_date, end_date):
    try:
        # Convert dates to strings for yfinance
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = (end_date + timedelta(days=1)).strftime('%Y-%m-%d')
        
        df = yf.download(ticker, start=start_str, end=end_str, interval='1d')

        if df.empty:
            st.warning("No data fetched. Please try a different date range.")
            return None, None

        # Get company name
        try:
            stock_info = yf.Ticker(ticker)
            company_name = stock_info.info.get('longName', ticker)
        except:
            company_name = ticker

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
        return None, None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df.columns.values]
        df.columns = [col.split('_')[0] if '_' in col else col for col in df.columns]

    df.sort_index(inplace=True)

    if df.empty or len(df) < 30:
        st.warning("Not enough valid trading days. Try a longer date range to get at least 30 trading days.")

    return df, company_name

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

    return df

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

def create_interactive_chart(df, show_ema12=True, show_ema26=True):
    """Create interactive candlestick chart with technical indicators"""
    
    # Create single plot for price data only
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Price",
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        )
    )
    
    # Add EMA 12
    if show_ema12 and 'EMA_12' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['EMA_12'],
                mode='lines',
                name='EMA 12',
                line=dict(color='#ff7f0e', width=2)
            )
        )
    
    # Add EMA 26
    if show_ema26 and 'EMA_26' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['EMA_26'],
                mode='lines',
                name='EMA 26',
                line=dict(color='#d62728', width=2)
            )
        )
    
    # Update layout
    fig.update_layout(
        title="Stock Price Analysis with Technical Indicators",
        xaxis_title="Date",
        yaxis_title="Closing Price",
        height=600,
        showlegend=True,
        template="plotly_white",
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="date",
            range=[df.index.min(), df.index.max()]  # Ensure full range is shown
        ),
        margin=dict(l=20, r=20, t=40, b=20)  # Adjust margins if needed
    )
    
    return fig

#streamlit
# Only enable button if validation passes
fetch_button_disabled = not date_validation_passed
if st.sidebar.button("Get Data & Run XGBoost", disabled=fetch_button_disabled):
    df, company_name = fetch_stock_data(ticker, start_date, end_date)

    if df is not None and not df.empty:
        st.success(f"Data fetched for {company_name} from {start_date} to {end_date}")
        st.dataframe(df.head(10))
        st.dataframe(df.tail(10))

        df1 = add_technical_indicators(df)
        df1.fillna(0, inplace=True)  
        st.subheader("Technical Indicators")
        st.dataframe(df1.head(10))
        st.dataframe(df1.tail(10))

        df = add_technical_indicators(df)
        df.dropna(inplace=True)
        df = prepare_features_for_regression(df)
        st.subheader("Features for Regression")
        st.dataframe(df.head(10))
        st.dataframe(df.tail(10))

    else:
        st.warning("No data found for this ticker or date range.")