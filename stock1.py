import streamlit as st
import holidays
import yfinance as yf
import pandas as pd
import numpy as np
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

default_end_date = datetime.today().date()
end_date = st.sidebar.date_input("Select End Date (last day of training data)", value=default_end_date)

# Automatically calculate start_date using a fixed window
# Use 270 calendar days to ensure ~180 valid trading days
start_date = end_date - timedelta(days=270)
st.sidebar.info(f"Using last 270 calendar days (from {start_date} to {end_date}) for model training")
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

    # NEW ENGINEERED FEATURES
    df['DailyRange'] = df['High'] - df['Low']
    df['PriceGap'] = df['Close'] - df['Open']
    df['PositionInRange'] = (df['Close'] - df['Low']) / (df['High'] - df['Low']).replace(0, np.nan)
    df['VolumeRatio_5'] = df['Volume'] / df['Volume'].rolling(window=5).mean()
    df['Volatility_5d'] = df['Close'].rolling(window=5).std()
    df['IsBullish'] = (df['Close'] > df['Open']).astype(int)

    # TARGETS
    for horizon in range(1, 4):
        df[f'Target_Close_{horizon}d'] = df['Close'].shift(-horizon)

    return df

nse_holidays = holidays.India(years=range(2020, 2035))  # Wider range if needed

def get_next_trading_day(date):
    next_day = date + pd.Timedelta(days=1)
    while next_day.weekday() >= 5 or next_day.date() in nse_holidays:
        next_day += pd.Timedelta(days=1)
    return next_day

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

def train_xgboost_model(df, end_date, n_days=1):
    df = df.copy()
    
    feature_cols = [
    'Open', 'High', 'Low', 'Close', 'Volume', 
    'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal', 
    'LogReturn_1d', 'BB_upper', 'BB_middle', 'BB_lower',
    'DailyRange', 'PriceGap', 'PositionInRange',
    'VolumeRatio_5', 'Volatility_5d', 'IsBullish']

    df.dropna(subset=feature_cols, inplace=True)
    df['Target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)

    X = df[feature_cols]
    y = df['Target']

    # TimeSeriesSplit preserves order
    tscv = TimeSeriesSplit(n_splits=5)

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
    }

    xgb_base = xgb.XGBRegressor(random_state=42)
    grid_search = GridSearchCV(
        xgb_base,
        param_grid,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_

    # Train-test split for final evaluation
    split_idx = int(len(X) * 0.9)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    accuracy = 100 - (mape * 100)

    st.markdown(f"""
    **🔧 Best Parameters Chosen:** `{grid_search.best_params_}`  
    **📊 Model Evaluation on Test Set:**  
    - MAE: `{mae:.4f}`  
    - RMSE: `{rmse:.4f}`  
    - R²: `{r2:.4f}`  
    - Accuracy (100 - MAPE): `{accuracy:.2f}%`
    """)

    # Forecasting
    if isinstance(end_date, datetime):
        end_date_ts = pd.Timestamp(end_date).tz_localize('Asia/Kolkata')
    else:
        end_date_ts = pd.Timestamp(end_date).tz_localize('Asia/Kolkata')

    forecast_start = get_next_trading_day(end_date_ts)

    future_dates = []
    predictions = []

    last_row = df.iloc[-1].copy()

    while len(predictions) < n_days:
        input_features = last_row[feature_cols].copy()
        pred_close = best_model.predict(input_features.values.reshape(1, -1))[0]

        predictions.append(pred_close)
        future_dates.append(forecast_start)

        last_row['Close'] = pred_close
        last_row['LogReturn_1d'] = np.log(pred_close / input_features['Close']) if input_features['Close'] != 0 else 0

        forecast_start = get_next_trading_day(forecast_start)

    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Close': predictions
    }).set_index('Date')

    return best_model, X_test, y_test, y_pred, forecast_df


#streamlit
# Only enable button if validation passes
fetch_button_disabled = not date_validation_passed
if st.sidebar.button("Get Data & Run XGBoost", disabled=fetch_button_disabled):
    df, company_name = fetch_stock_data(ticker, start_date, end_date)

    if df is not None and not df.empty:
        st.success(f"Data fetched for {company_name} from {start_date} to {end_date}")
        df = add_technical_indicators(df)
        df = prepare_features_for_regression(df)
        df1 = prepare_features_for_regression(df) 

        df.dropna(inplace=True)
        st.subheader("📊 XGBoost Model Training & Forecast")
        model, X_test, y_test, y_pred, forecast_df = train_xgboost_model(df, end_date)

        st.write("Model trained on past data. Here's how it performed on recent test data:")

        # Use get_next_trading_day to get actual next trading day for each test sample
        test_dates = pd.Series(X_test.index).apply(get_next_trading_day)

        results_df = pd.DataFrame({
                'Actual': y_test.values,
                'Predicted': y_pred
            }, index=test_dates)
    
        st.dataframe(results_df.tail(10))

        st.subheader("🔮 Stock Price Forecasting:")
        st.dataframe(forecast_df)

        # Prepare data for download
        df_processed = df1.fillna(0)
        
        # Create columns for download button and view option
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Create download button for preprocessed data
            csv_data = df_processed.to_csv(index=True).encode('utf-8')
            st.download_button(
                label="📥 Download Preprocessed Data (CSV)",
                data=csv_data,
                file_name=f"{ticker}_preprocessed_data_upto_{end_date}.csv",
                mime="text/csv",
                help="Download the complete preprocessed dataset with technical indicators and features"
            )
        
        with col2:
            # Option to view sample data using expander
            with st.expander("👁️ View Sample Data"):
                st.subheader("Sample of Preprocessed Data (Last 10 rows)")
                st.dataframe(df_processed.tail(10))
                st.info(f"Showing last 10 rows of {len(df_processed)} total rows with {len(df_processed.columns)} features")

    else:
        st.warning("No data found for this ticker or date range.")
