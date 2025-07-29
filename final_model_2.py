import streamlit as st
import holidays
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set page config
st.set_page_config(page_title="Stock App", layout="wide")

# Title
st.title("Stock Market Dashboard")

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

start_date = end_date - timedelta(days=270)

# Show info message
st.sidebar.info(
    f"Using last 270 calendar days (from {start_date} to {end_date}) for model training"
)
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

    # Relative Volume (10-day)
    df['Relative_Volume_10d'] = df['Volume'] / df['Volume'].rolling(10).mean().shift(1)

    # Gap % (Open vs Previous Close)
    df['Gap_Percent'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1) * 100

    # 5-day Momentum
    df['Momentum_5d'] = df['Close'] - df['Close'].rolling(5).mean().shift(1)

    # Sector Index Return
    # Map stock to sector index
    ticker_upper = str(df.attrs.get('ticker', '')).upper() if hasattr(df, 'attrs') else ''
    if 'TCS' in ticker_upper or 'INFY' in ticker_upper:
        sector_ticker = '^CNXIT'  # NIFTY IT
    elif 'RELIANCE' in ticker_upper:
        sector_ticker = '^CNXENERGY'  # NIFTY Energy
    else:
        sector_ticker = '^NSEI'  # NIFTY 50 as fallback
    try:
        sector_df = yf.download(sector_ticker, start=df.index.min(), end=df.index.max() + pd.Timedelta(days=1), interval='1d')
        sector_df.index = pd.to_datetime(sector_df.index).tz_localize(None)
        # Align sector index to stock dates
        sector_df = sector_df.reindex(df.index, method='ffill')
        df['Sector_Return'] = (sector_df['Close'] - sector_df['Close'].shift(1)) / sector_df['Close'].shift(1) * 100
    except Exception as e:
        df['Sector_Return'] = 0  # fallback if sector data not available

    return df, company_name

# Apply Technical indicators 
def add_technical_indicators(df):
    df = df.copy()
    
    #EMA12 and EMA26
    df['EMA_12'] = EMAIndicator(close=df['Close'], window=12).ema_indicator()
    df['EMA_26'] = EMAIndicator(close=df['Close'], window=26).ema_indicator()

    # MACD
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()

    # RSI
    df['RSI_14'] = RSIIndicator(close=df['Close'], window=14).rsi()

    # Bollinger Bands
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_lower'] = bb.bollinger_lband()
    df['BB_middle'] = bb.bollinger_mavg()

    # ADX
    adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    df['ADX_14'] = adx.adx()

    df.dropna(inplace=True)
    return df

#Feature engineering 
def prepare_features_for_regression(df):
    df = df.copy()

    # Calendar-Time Features
    df['DayOfWeek'] = df.index.dayofweek  # 0 = Monday, 4 = Friday
    df['Month'] = df.index.month

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
    df['Return_5d'] = df['Close'].pct_change(5)
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

    # Bollinger Band Position (%)
    df['Bollinger_Position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower']).replace(0, np.nan)

    # 10-day Range Position
    rolling_max = df['Close'].rolling(window=10).max()
    rolling_min = df['Close'].rolling(window=10).min()
    df['RangePos_10d'] = (df['Close'] - rolling_min) / (rolling_max - rolling_min).replace(0, np.nan)

    # 🔧 VOLATILITY FEATURES (NEW)
    # Feature 1: Yesterday's % Price Change
    df['Yesterday_Price_Change'] = df['Close'].pct_change(1)
    
    # Feature 2: 5-Day Rolling Volatility (standard deviation of returns)
    df['Rolling_Volatility_5d'] = df['Return_1d'].rolling(window=5).std()
    
    # Feature 3: ATR (Average True Range) - 14-day period
    # True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
    df['True_Range'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            np.abs(df['High'] - df['Close'].shift(1)),
            np.abs(df['Low'] - df['Close'].shift(1))
        )
    )
    df['ATR_14d'] = df['True_Range'].rolling(window=14).mean()

    df.dropna(inplace=True) 
    return df

nse_holidays = holidays.India(years=range(2020, 2035))  # Wider range if needed

def get_next_trading_day(date):
    next_day = date + pd.Timedelta(days=1)
    while next_day.weekday() >= 5 or next_day.date() in nse_holidays:
        next_day += pd.Timedelta(days=1)
    return next_day

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

def train_xgboost_classifier(df, end_date, n_days=1):
    df = df.copy()
    
    feature_cols = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal',
    'BB_upper', 'BB_middle', 'BB_lower',
    'Bollinger_Position', 'RangePos_10d',
    'RSI_14', 'ADX_14',
    'RollingMean_3', 'RollingStd_3', 'RollingMean_7', 'RollingStd_7',
    'Return_1d', 'Return_5d', 'LogReturn_1d',
    'VolumeRatio_5','Volatility_5d',
    'DailyRange', 'PriceGap', 'PositionInRange',
    'Close_lag_1', 'Close_lag_2', 'Close_lag_3',
    'Volume_lag_1', 'Volume_lag_2', 'Volume_lag_3',
    'DayOfWeek', 'Month', 'IsBullish',
    # 🔧 NEW VOLATILITY FEATURES
    'Yesterday_Price_Change', 'Rolling_Volatility_5d', 'ATR_14d',
    'Relative_Volume_10d', 'Gap_Percent', 'Sector_Return', 'Momentum_5d']

    df.dropna(subset=feature_cols, inplace=True)
    
    # 🔧 STEP 1: CHANGE THE TARGET - Classification instead of regression
    # Create binary target: 1 if price goes UP, 0 if DOWN
    df['Price_Next_Day'] = df['Close'].shift(-1)
    df['Target'] = (df['Price_Next_Day'] > df['Close']).astype(int)  # 1 for UP, 0 for DOWN
    df.dropna(inplace=True)

    X = df[feature_cols]
    y = df['Target']

    # TimeSeriesSplit preserves order
    tscv = TimeSeriesSplit(n_splits=5)

    param_grid = {
        "n_estimators": [400, 600, 800],
        "max_depth": [6, 7],
        "learning_rate": [0.02, 0.03],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
    }

    # MODEL TYPE - XGBoost Classifier
    xgb_base = xgb.XGBClassifier(random_state=42)
    grid_search = GridSearchCV(
        xgb_base,
        param_grid,
        cv=tscv,
        scoring='accuracy',  
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_

    # Train-test split for final evaluation with validation set
    split_idx = int(len(X) * 0.8)  # Use 80% for training
    val_split_idx = int(len(X) * 0.9)  # Use 10% for validation
    X_train = X.iloc[:split_idx]
    X_val = X.iloc[split_idx:val_split_idx]
    X_test = X.iloc[val_split_idx:]
    y_train = y.iloc[:split_idx]
    y_val = y.iloc[split_idx:val_split_idx]
    y_test = y.iloc[val_split_idx:]

    # Check if we have enough data for validation
    if len(X_val) < 5:
        st.warning("⚠️ Not enough data for validation set. Using simpler training approach.")
        # Fall back to simple training without early stopping
        best_model.fit(X_train, y_train)
    else:
        # Train without early stopping
        best_model.fit(X_train, y_train)
    
    # 🔧 STEP 4: CHANGE EVALUATION METRICS - Classification metrics instead of regression
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]  # Probability of UP class

    # --- CONFIDENCE THRESHOLDING LOGIC ---
    threshold_up = 0.65
    threshold_down = 0.35
    thresholded_pred = []
    thresholded_conf = []
    for prob in y_pred_proba:
        if prob > threshold_up:
            thresholded_pred.append(1)
            thresholded_conf.append(prob)
        elif prob < threshold_down:
            thresholded_pred.append(0)
            thresholded_conf.append(prob)
        else:
            thresholded_pred.append(None)  # Uncertain
            thresholded_conf.append(prob)

    # Only keep confident predictions for metrics
    confident_indices = [i for i, p in enumerate(thresholded_pred) if p is not None]
    y_test_conf = y_test.iloc[confident_indices]
    y_pred_conf = [thresholded_pred[i] for i in confident_indices]
    y_conf_proba = [thresholded_conf[i] for i in confident_indices]

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    if len(y_pred_conf) > 0:
        accuracy_conf = accuracy_score(y_test_conf, y_pred_conf) * 100
        precision_conf = precision_score(y_test_conf, y_pred_conf, zero_division=0) * 100
        recall_conf = recall_score(y_test_conf, y_pred_conf, zero_division=0) * 100
        f1_conf = f1_score(y_test_conf, y_pred_conf, zero_division=0) * 100
    else:
        accuracy_conf = precision_conf = recall_conf = f1_conf = 0.0

    percent_confident = 100 * len(y_pred_conf) / len(y_test) if len(y_test) > 0 else 0

    # --- END CONFIDENCE THRESHOLDING LOGIC ---

    # 🔧 STEP 4: UPDATE EVALUATION DISPLAY - Classification metrics
    # Removed evaluation metrics and confusion matrix display here to move to main app section

    # After model predictions and metrics
    # Compute naive predictions (simple rule: if yesterday was up, predict up)
    naive_pred = (X_test['Return_1d'] > 0).astype(int)  # Naive: if yesterday was up, predict up
    y_test_actual = y_test.values
    
    # Metrics for naive model
    accuracy_naive = accuracy_score(y_test_actual, naive_pred) * 100
    precision_naive = precision_score(y_test_actual, naive_pred, zero_division=0) * 100
    recall_naive = recall_score(y_test_actual, naive_pred, zero_division=0) * 100
    f1_naive = f1_score(y_test_actual, naive_pred, zero_division=0) * 100

    # Display comparison as a DataFrame for better side-by-side view
    metrics_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'XGBoost': [f'{accuracy_conf:.1f}%', f'{precision_conf:.1f}%', f'{recall_conf:.1f}%', f'{f1_conf:.1f}%'],
        'Naive': [f'{accuracy_naive:.1f}%', f'{precision_naive:.1f}%', f'{recall_naive:.1f}%', f'{f1_naive:.1f}%']
    }
    metrics_df = pd.DataFrame(metrics_data)
    # Removed Model vs. Naive Comparison display here

    # 🔧 STEP 5: UPDATE FORECASTING - Classification predictions instead of price predictions
    # Get prediction for next trading day
    last_row = df.iloc[-1].copy()
    input_features = last_row[feature_cols].copy()
    
    # Get prediction probability (probability of UP)
    pred_probability = best_model.predict_proba(input_features.values.reshape(1, -1))[0][1]
    pred_direction = best_model.predict(input_features.values.reshape(1, -1))[0]
    
    # Determine confidence level and prediction
    if pred_probability >= 0.65:
        confidence_level = "HIGH"
        trend_emoji = "📈" if pred_direction == 1 else "📉"
    elif pred_probability >= 0.55:
        confidence_level = "MEDIUM"
        trend_emoji = "📈" if pred_direction == 1 else "📉"
    else:
        confidence_level = "LOW"
        trend_emoji = "🤔"
    
    # Create forecast display
    predicted_trend = "UP" if pred_direction == 1 else "DOWN"
    forecast_result = {
        'Predicted_Trend': predicted_trend,
        'Confidence': f"{pred_probability:.1%}",
        'Confidence_Level': confidence_level,
        'Recommendation': f"{trend_emoji} {predicted_trend} with {confidence_level} confidence"
    }

    return best_model, X_test, y_test, y_pred, forecast_result, thresholded_pred, thresholded_conf, pred_probability

# LSTM Model Definition (from lstm.py)
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 64)
        c0 = torch.zeros(2, x.size(0), 64)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

# LSTM Training and Prediction Function

def run_lstm_regression(df, seq_len=60, epochs=100):
    df = df.copy()
    # Remove target columns used for classifier
    drop_cols = [col for col in df.columns if col.startswith('Target_Close_') or col == 'Target']
    feature_cols = [col for col in df.columns if col not in drop_cols]
    # Target: next-day raw return
    df['target_return'] = (df['Close'].shift(-1) / df['Close']) - 1
    df = df.dropna()
    feature_cols = [col for col in feature_cols if col in df.columns and col != 'target_return']
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(df[feature_cols])
    target = df['target_return'].values
    X, y = [], []
    for i in range(seq_len, len(df)):
        X.append(features_scaled[i-seq_len:i])
        y.append(target[i])
    if len(X) == 0:
        return None, None, None, None
    X = np.array(X)
    y = np.array(y)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    model = LSTM(input_size=X.shape[2])
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        pred = model(X_tensor)
        loss = loss_fn(pred, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        y_pred = model(X_tensor).numpy()
    predicted_return = y_pred[-1][0]
    last_price = df['Close'].iloc[-1]
    predicted_price = last_price * (1 + predicted_return)
    return predicted_return, predicted_price, y, y_pred

#streamlit
# Only enable button if validation passes
fetch_button_disabled = not date_validation_passed
if st.sidebar.button("Get Predictions", disabled=fetch_button_disabled):
    df, company_name = fetch_stock_data(ticker, start_date, end_date)

    if df is not None and not df.empty:
        st.success(f"Data fetched for {company_name} from {start_date} to {end_date}")
        df = add_technical_indicators(df)
        df = prepare_features_for_regression(df)
        df1 = prepare_features_for_regression(df) 

        df.dropna(inplace=True)

        # --- XGBoost Classifier ---
        model, X_test, y_test, y_pred, forecast_result, thresholded_pred, thresholded_conf, pred_probability = train_xgboost_classifier(df, end_date)

        # Use get_next_trading_day to get actual next trading day for each test sample
        test_dates = pd.Series(X_test.index).apply(get_next_trading_day)
        # Create results DataFrame with actual vs predicted directions and confidence
        actual_directions = ["UP" if x == 1 else "DOWN" for x in y_test.values]
        predicted_directions = []
        confidence_vals = []
        thresholded_labels = []
        for pred, prob in zip(thresholded_pred, thresholded_conf):
            if pred == 1:
                predicted_directions.append("UP")
                thresholded_labels.append("UP")
            elif pred == 0:
                predicted_directions.append("DOWN")
                thresholded_labels.append("DOWN")
            else:
                predicted_directions.append("UNCERTAIN")
                thresholded_labels.append("UNCERTAIN")
            confidence_vals.append(f"{prob:.2f}")
        results_df = pd.DataFrame({
            'Actual_Direction': actual_directions,
            'Predicted_Direction': predicted_directions,
            'Confidence': confidence_vals,
            'Thresholded_Prediction': thresholded_labels,
            'Correct': [(a == p) if p != "UNCERTAIN" else None for a, p in zip(actual_directions, thresholded_labels)]
        }, index=test_dates)
        
        st.subheader("🔮 Stock Price Direction Forecast using XGBoost Classifier")
        # Display all forecast elements 
        st.markdown(f"""
        **Predicted Trend:** {forecast_result['Predicted_Trend']}  
        **Model Confidence:** {forecast_result['Confidence']}  
        **Confidence Level:** {forecast_result['Confidence_Level']}  
        **Recommendation:** {forecast_result['Recommendation']}
        """)

        # Add evaluation metrics and confusion matrix here after forecast
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        # Calculate evaluation metrics for confident predictions
        confident_indices = [i for i, p in enumerate(thresholded_pred) if p is not None]
        y_test_conf = y_test.iloc[confident_indices]
        y_pred_conf = [thresholded_pred[i] for i in confident_indices]
        if len(y_pred_conf) > 0:
            accuracy_conf = accuracy_score(y_test_conf, y_pred_conf) * 100
            precision_conf = precision_score(y_test_conf, y_pred_conf, zero_division=0) * 100
            recall_conf = recall_score(y_test_conf, y_pred_conf, zero_division=0) * 100
            f1_conf = f1_score(y_test_conf, y_pred_conf, zero_division=0) * 100
        else:
            accuracy_conf = precision_conf = recall_conf = f1_conf = 0.0
        percent_confident = 100 * len(y_pred_conf) / len(y_test) if len(y_test) > 0 else 0
        
        st.markdown(f""" 
        Model Evaluation on Test Set (Confident Predictions Only):
        - Accuracy: `{accuracy_conf:.2f}%`  
        - Precision: `{precision_conf:.2f}%`  
        - Recall: `{recall_conf:.2f}%`  
        - F1-Score: `{f1_conf:.2f}%`  
        - Predictions made on `{len(y_pred_conf)}` out of `{len(y_test)}` days (`{percent_confident:.1f}%`)
        """)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        st.markdown(f"""
        Confusion Matrix:
        - True Positives (UP predicted as UP): `{cm[1,1]}`
        - False Positives (DOWN predicted as UP): `{cm[0,1]}`
        - False Negatives (UP predicted as DOWN): `{cm[1,0]}`
        - True Negatives (DOWN predicted as DOWN): `{cm[0,0]}`
        """)

        if len(y_pred_conf) < len(y_test):
            st.info(f"Model skipped {len(y_test) - len(y_pred_conf)} days due to low confidence.")
    
        # --- LSTM Integration ---
        st.subheader("Next-Day Price Prediction using LSTM Regressor")
        with st.spinner("Training LSTM model..."):
            predicted_return, predicted_price, y_lstm, y_pred_lstm = run_lstm_regression(df)
            if predicted_return is not None:
                st.success(f"LSTM Predicted Next-Day Close as ₹{predicted_price:.2f}")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.plot(y_lstm, label='Actual Return')
                ax.plot(y_pred_lstm, label='Predicted Return')
                ax.legend()
                st.pyplot(fig)
                # 📊 LSTM Evaluation Metrics
 

                mae_lstm = mean_absolute_error(y_lstm, y_pred_lstm)
                mse_lstm = mean_squared_error(y_lstm, y_pred_lstm)
                rmse_lstm = np.sqrt(mse_lstm)
                r2_lstm = r2_score(y_lstm, y_pred_lstm)

                # Create a DataFrame with actual and predicted close prices
                with st.expander("📅Predictions Table "):
                    recent_dates = df.tail(len(y_pred_lstm)).index[-15:]  # last 15 dates
                    recent_actual = df['Close'].values[-len(y_pred_lstm):][-15:]  # actual close
                    prev_close = df['Close'].shift(1).values[-len(y_pred_lstm):][-15:]  # previous close
                    predicted_returns = y_pred_lstm[-15:].flatten()  # predicted return
                    predicted_closes = prev_close * (1 + predicted_returns)  # predicted close

                    result_table = pd.DataFrame({
                    'Date': recent_dates,
                    'Prev Close': np.round(prev_close, 2),
                    'Actual Close': np.round(recent_actual, 2),
                    'Predicted Close': np.round(predicted_closes, 2)
                    })

                    st.dataframe(result_table.set_index("Date"))


                # Directional Accuracy Calculation
                actual_direction = np.sign(y_lstm)
                predicted_direction = np.sign(y_pred_lstm)
                accuracy = np.mean(actual_direction == predicted_direction) * 100


                st.markdown("#### 📈 LSTM Evaluation Metrics on Historical Data")
                st.markdown(f"""
                - **MAE (Mean Absolute Error):** `{mae_lstm:.6f}`  
                - **RMSE (Root Mean Squared Error):** `{rmse_lstm:.6f}`  
                - **R² Score:** `{r2_lstm:.4f}
                - **Accuracy:** `{accuracy:.2f}%`
                """)
                # Display Directional Accuracy
                #st.markdown(f"**Accuracy:** `{accuracy:.2f}%`")

            else:
                st.warning("Not enough data for LSTM model.")
     
        
        # Prepare data for download
        df_processed = df1.fillna(0)
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
