import streamlit as st
import holidays
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

# Set page config
st.set_page_config(page_title="Stock App", layout="wide")

# Title
st.title("📈 Stock Market Dashboard")

from datetime import datetime, date, timedelta
import streamlit as st

# Sidebar Inputs
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., TCS.NS)", value="TCS.NS")

today = datetime.today().date()
# Let user select end date, but restrict to today or earlier
end_date = st.sidebar.date_input(
    "Select End Date (last day of training data)",
    value=today,
    max_value=today
)

# Defensive check (in case someone hacks the URL to pass future dates)
if end_date > today:
    st.sidebar.error("❌ End date cannot be in the future.")
    st.stop()

# Fetch a large window of data first, then filter to 500 trading days
raw_start_date = end_date - timedelta(days=1000)  # Fetch more than 500 trading days to be safe

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

# Apply Technical indicators 
def add_technical_indicators(df):
    df = df.copy()
    # EMA-10, EMA-50, SMA-200
    df['EMA_10'] = EMAIndicator(close=df['Close'], window=10).ema_indicator()
    df['EMA_50'] = EMAIndicator(close=df['Close'], window=50).ema_indicator()
    df['SMA_200'] = SMAIndicator(close=df['Close'], window=200).sma_indicator()

    # MACD histogram only
    macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD_hist'] = macd.macd_diff()

    # RSI-14, clipped
    rsi = RSIIndicator(close=df['Close'], window=14).rsi()
    df['RSI_14'] = rsi.clip(5, 95)

    # ATR percent (14)
    atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()
    df['ATR_pct_14'] = (atr / df['Close']).clip(0, 0.2)

    # Bollinger %B (BB_percent)
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    bb_percent = (df['Close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
    df['BB_percent'] = bb_percent

    # Volume z-score (30d), clipped
    vol_mean = df['Volume'].rolling(window=30).mean()
    vol_std = df['Volume'].rolling(window=30).std()
    df['Volume_z30'] = ((df['Volume'] - vol_mean) / vol_std).clip(-3, 3)

    # Calendar features
    df['DayOfWeek'] = df.index.dayofweek  # 0=Monday
    df['Month'] = df.index.month
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

    # Shift all indicators by +1 to prevent leakage
    indicator_cols = [
        'EMA_10', 'EMA_50', 'SMA_200', 'MACD_hist', 'RSI_14',
        'ATR_pct_14', 'BB_percent', 'Volume_z30',
        'DayOfWeek', 'Month_sin', 'Month_cos'
    ]
    df[indicator_cols] = df[indicator_cols].shift(1)

    # Drop initial NaNs after shifting (max window = 200)
    df = df.iloc[200:]
    return df

#Feature engineering 
def prepare_features_for_regression(df):
    df = df.copy()
    # 1. Keep technical indicator columns
    feature_cols = [
        'EMA_10', 'EMA_50', 'SMA_200', 'MACD_hist', 'RSI_14',
        'ATR_pct_14', 'BB_percent', 'Volume_z30',
        'DayOfWeek', 'Month_sin', 'Month_cos'
    ]
    # 2. Add lagged closing prices (1, 2, 3 days ago)
    df['Close_lag_1'] = df['Close'].shift(1)
    df['Close_lag_2'] = df['Close'].shift(2)
    df['Close_lag_3'] = df['Close'].shift(3)
    feature_cols += ['Close_lag_1', 'Close_lag_2', 'Close_lag_3']
    # Add daily log return
    df['LogReturn_1d'] = np.log(df['Close'] / df['Close'].shift(1))
    feature_cols.append('LogReturn_1d')
    # Add daily range (High-Low)/Close
    df['Range_1d'] = (df['High'] - df['Low']) / df['Close']
    feature_cols.append('Range_1d')
    # 3. Target: next day's log return
    df['Target'] = np.log(df['Close'].shift(-1) / df['Close'])
    # 4. Drop rows with missing or infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols + ['Target'])
    # 5. Limit to most recent 500 rows
    df = df.tail(500)
    return df, feature_cols

nse_holidays = holidays.India(years=range(2020, 2035))  # Wider range if needed

def get_next_trading_day(date):
    next_day = date + pd.Timedelta(days=1)
    while next_day.weekday() >= 5 or next_day.date() in nse_holidays:
        next_day += pd.Timedelta(days=1)
    return next_day

def train_xgboost_model(df, feature_cols):
    df = df.copy()
    X = df[feature_cols]
    y = df['Target']  # log-return target

    # Single 90/10 train/test split
    split_idx = int(len(X) * 0.9)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Hyperparameter search (GridSearchCV)
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
    }
    tscv = TimeSeriesSplit(n_splits=3)
    xgb_base = xgb.XGBRegressor(random_state=42)
    grid_search = GridSearchCV(
        xgb_base,
        param_grid,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Fit the best model ONCE on the training split
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    # Evaluate in log-return space
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    # Also compute training error for overfitting/underfitting check
    y_train_pred = best_model.predict(X_train)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    st.markdown(f"""
    **📊 Model Evaluation on Test Set (Log-Return):**  
    - MAE: `{mae:.4f}`  
    - RMSE: `{rmse:.4f}`  
    - R²: `{r2:.4f}`  
    - MAPE: `{mape:.4f}`
    - Training MAE: `{train_mae:.4f}`
    """)

    # Convert log-returns to closing prices for display
    # For each test point, previous day's close is needed
    prev_closes = X_test['Close_lag_1'].values
    actual_prices = prev_closes * np.exp(y_test.values)
    predicted_prices = prev_closes * np.exp(y_pred)

    # Display table of actual vs predicted closing prices
    price_df = pd.DataFrame({
        'Prev_Close': prev_closes,
        'Actual_Close': actual_prices,
        'Predicted_Close': predicted_prices
    }, index=X_test.index)
    st.subheader("📈 Actual vs. Predicted Closing Prices (Test Set)")
    st.dataframe(price_df.tail(30))

    # Line chart for last 30 test points
    st.line_chart(price_df[['Actual_Close', 'Predicted_Close']].tail(30))

    return best_model, X_test, y_test, y_pred


#streamlit
# Only enable button if validation passes
fetch_button_disabled = False
if st.sidebar.button("Get Data & Run XGBoost", disabled=fetch_button_disabled):
    df, company_name = fetch_stock_data(ticker, raw_start_date, end_date)

    if df is not None and not df.empty:
        st.success(f"Data fetched for {company_name} from {raw_start_date} to {end_date}")
        df = add_technical_indicators(df)
        df, feature_cols = prepare_features_for_regression(df)
        df1, _ = prepare_features_for_regression(df)

        # After fetching, filter to the last 500 trading days
        df = df.tail(500)
        start_date = df.index[0].date()

        st.subheader("📊 XGBoost Model Training & Forecast")
        model, X_test, y_test, y_pred = train_xgboost_model(df, feature_cols)

        st.write("Model trained on past data. Here's how it performed on recent test data:")

        # Use get_next_trading_day to get actual next trading day for each test sample
        test_dates = pd.Series(X_test.index).apply(get_next_trading_day)

        results_df = pd.DataFrame({
                'Actual': y_test.values,
                'Predicted': y_pred
            }, index=test_dates)
    
        st.dataframe(results_df.tail(10))

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