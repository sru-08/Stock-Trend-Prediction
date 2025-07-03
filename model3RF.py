import streamlit as st
import holidays
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Set page config
st.set_page_config(page_title="Stock App", layout="wide")

# Title
st.title("📈 Stock Market Dashboard")

# Sidebar Inputs
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., TCS.NS)", value="TCS.NS")
today = datetime.today().date()
end_date = st.sidebar.date_input(
    "Select End Date (last day of training data)",
    value=today,
    max_value=today
)
if end_date > today:
    st.sidebar.error("❌ End date cannot be in the future.")
    st.stop()
raw_start_date = end_date - timedelta(days=1000)

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

# Feature engineering for Random Forest

def prepare_rf_features(df):
    df = df.copy()
    # Technical indicators
    df['EMA_10'] = EMAIndicator(close=df['Close'], window=10).ema_indicator()
    df['EMA_50'] = EMAIndicator(close=df['Close'], window=50).ema_indicator()
    df['SMA_200'] = SMAIndicator(close=df['Close'], window=200).sma_indicator()
    macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD_hist'] = macd.macd_diff()
    df['RSI_14'] = RSIIndicator(close=df['Close'], window=14).rsi().clip(5, 95)
    atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()
    df['ATR_pct_14'] = (atr / df['Close']).clip(0, 0.2)
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_percent'] = (df['Close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
    vol_mean = df['Volume'].rolling(window=30).mean()
    vol_std = df['Volume'].rolling(window=30).std()
    df['Volume_z30'] = ((df['Volume'] - vol_mean) / vol_std).clip(-3, 3)
    # Price-based features
    for lag in [1,2,3,5,10]:
        df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
        df[f'Return_lag_{lag}'] = (df['Close'] / df['Close'].shift(lag)) - 1
    df['RollingMean_5'] = df['Close'].rolling(window=5).mean()
    df['RollingStd_5'] = df['Close'].rolling(window=5).std()
    df['RollingMean_20'] = df['Close'].rolling(window=20).mean()
    df['RollingStd_20'] = df['Close'].rolling(window=20).std()
    df['Range_1d'] = (df['High'] - df['Low']) / df['Close']
    # Calendar features
    df['DayOfWeek'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    # Targets
    df['Target_return'] = (df['Close'].shift(-1) / df['Close']) - 1
    df['Target_direction'] = (df['Target_return'] > 0).astype(int)
    # Drop NaNs
    feature_cols = [
        'EMA_10', 'EMA_50', 'SMA_200', 'MACD_hist', 'RSI_14', 'ATR_pct_14', 'BB_percent', 'Volume_z30',
        'Close_lag_1', 'Close_lag_2', 'Close_lag_3', 'Close_lag_5', 'Close_lag_10',
        'Return_lag_1', 'Return_lag_2', 'Return_lag_3', 'Return_lag_5', 'Return_lag_10',
        'RollingMean_5', 'RollingStd_5', 'RollingMean_20', 'RollingStd_20', 'Range_1d',
        'DayOfWeek', 'Month_sin', 'Month_cos'
    ]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols + ['Target_return', 'Target_direction'])
    df = df.tail(500)
    return df, feature_cols

# Main logic
fetch_button_disabled = False
if st.sidebar.button("Get Data & Run Random Forest", disabled=fetch_button_disabled):
    df, company_name = fetch_stock_data(ticker, raw_start_date, end_date)
    if df is not None and not df.empty:
        st.success(f"Data fetched for {company_name} from {raw_start_date} to {end_date}")
        df = add_technical_indicators(df)  # Optionally keep for UI/sample
        df_rf, feature_cols = prepare_rf_features(df)
        # Train/test split
        split_idx = int(len(df_rf) * 0.8)
        X = df_rf[feature_cols]
        y_reg = df_rf['Target_return']
        y_clf = df_rf['Target_direction']
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_reg_train, y_reg_test = y_reg.iloc[:split_idx], y_reg.iloc[split_idx:]
        y_clf_train, y_clf_test = y_clf.iloc[:split_idx], y_clf.iloc[split_idx:]
        # Regression
        rf_reg = RandomForestRegressor(random_state=42)
        rf_reg.fit(X_train, y_reg_train)
        y_reg_pred = rf_reg.predict(X_test)
        # Classification
        rf_clf = RandomForestClassifier(random_state=42)
        rf_clf.fit(X_train, y_clf_train)
        y_clf_pred = rf_clf.predict(X_test)
        # Regression metrics
        mae = mean_absolute_error(y_reg_test, y_reg_pred)
        rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
        r2 = r2_score(y_reg_test, y_reg_pred)
        mape = mean_absolute_percentage_error(y_reg_test, y_reg_pred)
        st.markdown(f"""
        **📊 Random Forest Regression (Raw Return) Test Set:**  
        - MAE: `{mae:.4f}`  
        - RMSE: `{rmse:.4f}`  
        - R²: `{r2:.4f}`  
        - MAPE: `{mape:.4f}`
        """)
        # Actual vs predicted closing prices
        prev_closes = X_test['Close_lag_1'].values
        actual_prices = prev_closes * (1 + y_reg_test.values)
        predicted_prices = prev_closes * (1 + y_reg_pred)
        price_df = pd.DataFrame({
            'Prev_Close': prev_closes,
            'Actual_Close': actual_prices,
            'Predicted_Close': predicted_prices
        }, index=X_test.index)
        st.subheader("📈 Actual vs. Predicted Closing Prices (Test Set)")
        st.dataframe(price_df.tail(30))
        st.line_chart(price_df[['Actual_Close', 'Predicted_Close']].tail(30))
        # Classification metrics
        acc = accuracy_score(y_clf_test, y_clf_pred)
        prec = precision_score(y_clf_test, y_clf_pred)
        rec = recall_score(y_clf_test, y_clf_pred)
        f1 = f1_score(y_clf_test, y_clf_pred)
        st.markdown(f"""
        **📊 Random Forest Classification (Direction) Test Set:**  
        - Accuracy: `{acc:.4f}`  
        - Precision: `{prec:.4f}`  
        - Recall: `{rec:.4f}`  
        - F1: `{f1:.4f}`
        """)
        cm = confusion_matrix(y_clf_test, y_clf_pred)
        st.write("Confusion Matrix:")
        st.dataframe(pd.DataFrame(cm, columns=["Predicted Down", "Predicted Up"], index=["Actual Down", "Actual Up"]))
        # Prepare data for download and view
        df_processed = df_rf.fillna(0)
        col1, col2 = st.columns([1, 1])
        with col1:
            csv_data = df_processed.to_csv(index=True).encode('utf-8')
            st.download_button(
                label="📥 Download Preprocessed Data (CSV)",
                data=csv_data,
                file_name=f"{ticker}_preprocessed_data_upto_{end_date}.csv",
                mime="text/csv",
                help="Download the complete preprocessed dataset with technical indicators and features"
            )
        with col2:
            with st.expander("👁️ View Sample Data"):
                st.subheader("Sample of Preprocessed Data (Last 10 rows)")
                st.dataframe(df_processed.tail(10))
                st.info(f"Showing last 10 rows of {len(df_processed)} total rows with {len(df_processed.columns)} features")
    else:
        st.warning("No data found for this ticker or date range.")