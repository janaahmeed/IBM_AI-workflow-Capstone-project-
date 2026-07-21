import os
import logging
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error 
import json 
from datetime import datetime
import ollama


#Logging 

logging.basicConfig(
    filename="app_performance.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = Flask(__name__)
GLOBAL_MODEL = None


#  data ingestion FUNC

def ingest_data(file_path):
    
    if not os.path.exists(file_path):
        logging.error(f"Ingestion failed: File not found at {file_path}")
        raise FileNotFoundError(f"Data file not found at {file_path}")
    
    logging.info(f"Starting data ingestion from: {file_path}")
    df = pd.read_csv(file_path, low_memory=False)
    
    # Unify mismatched columns
    df['price'] = df['price'].fillna(df.get('total_price', np.nan))
    df['stream_id'] = df['stream_id'].fillna(df.get('StreamID', np.nan))
    df['times_viewed'] = df['times_viewed'].fillna(df.get('TimesViewed', np.nan))
    
    drop_cols = ['total_price', 'StreamID', 'TimesViewed']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors='ignore')
    
    df['date'] = pd.to_datetime( df[['year', 'month', 'day']])
    logging.info(f"Data successfully ingested. Shape: {df.shape}")
    return df


#feature engineering FUNC 

def engineer_features(df):
    logging.info("Starting feature engineering pipeline...")
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    
    daily_df = df.groupby('date').agg(
        total_value=('price', 'sum'),
        transaction_count=('price', 'count'),
        country_count=('country', 'nunique'),
        invoices_count=('invoice', 'count'),
        times_viewed=('times_viewed', 'sum')
    ).reset_index().sort_values(by='date')
    print("--- 1. Aggregating Daily Distribution Details ---")
    daily_df.to_csv("total_value_by_day.csv", index=False)
    print("Your file is saved exactly here:")
    print(os.path.abspath("Aggregating Daily Distribution Details.csv"))
    
    print("--- Country Performance Analysis: ---")
    total_revenue_by_Country = df.groupby('country').agg(
    total_value=('price', 'sum'),
    transaction_count=('price', 'count'),
    invoices_count=('invoice', 'count')
    ).reset_index().sort_values(by='total_value', ascending=False) 
    total_revenue_by_Country.to_csv("total_value_by_country.csv", index=False)
    
    min_date, max_date = df['date'].min(),df['date'].max()
    date_span = (max_date - min_date).days + 1
    print(f"\nEarliest Date: {min_date.date()} | Latest Date: {max_date.date()} | Total Range: {date_span} days")

    # Feature creation
    daily_df['rolling_mn_1Mon'] = daily_df['total_value'].rolling(window=30, min_periods=1).mean()
    daily_df['rolling_std_1Mon'] = daily_df['total_value'].rolling(window=30, min_periods=1).std()
    
    for lag in [30, 60, 90]:
        daily_df[f'lag_{lag}'] = daily_df['total_value'].shift(lag)
        
    daily_df['month'] = daily_df['date'].dt.month
    daily_df['quarter'] = daily_df['date'].dt.quarter
    
    # Target creation (90 days horizon)
    forecast_horizon = 90
    target_columns = []
    for step in range(1, forecast_horizon + 1):
        col_name = f'target_day_{step}'
        daily_df[col_name] = daily_df['total_value'].shift(-step)
        target_columns.append(col_name)
        
    daily_df.dropna(inplace=True)
    
    feature_cols = ['rolling_mn_1Mon', 'rolling_std_1Mon', 'lag_30', 'lag_60', 'lag_90', 'month', 'quarter']
    logging.info("Feature engineering complete.")
    return daily_df[feature_cols], daily_df[target_columns], daily_df

def train_production_model_RFR(df,X, Y):
    global GLOBAL_MODEL
    logging.info("Training production RandomForestRegressor model...")
    
#  TimeSeries Cross Validation 
    tscv = TimeSeriesSplit(n_splits=20)
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    scores = []

    print("\nRunning Cross-Validation Window Splits...")
    for fold, (train_index, test_index) in enumerate(tscv.split(X), 1):
      x_train, x_test = X.iloc[train_index], X.iloc[test_index]
      y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
    
      model.fit(x_train, y_train)
      preds = model.predict(x_test)
      rmse = root_mean_squared_error(y_test, preds)
      scores.append(rmse )
      print(f" Fold {fold} rmse: {rmse :.2f}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"rmse used RFR after splits {timestamp}.json"
    response_dict = {
        "Avg_rmse ":  sum(scores) / 20
    }
    with open(filename, 'w') as f:
        json.dump(response_dict, f, indent=4)
        
    logging.info(f"results saved locally to file: {filename}")
    
    print(response_dict)
    GLOBAL_MODEL = model
    logging.info("Model trained successfully and set to global variable.")
    return model


#Predicts a 90-day multi-point forecast sequence

def generate_executive_summary(predictions, payload_dict, model=None, feature_names=None, api_key=None):
    """
    Translates a 90-day prediction vector & feature importances into 
    an executive natural language summary.
    """
    preds = np.array(predictions)
    
    # 1. Compute Time-Series Vector Metrics 
    total_volume = float(np.sum(preds))
    avg_daily = float(np.mean(preds))
    peak_day = int(np.argmax(preds)) + 1
    peak_val = float(preds[peak_day - 1])
    
    # Trajectory calculation
    first_30_avg = np.mean(preds[:30])
    last_30_avg = np.mean(preds[-30:])
    growth_pct = ((last_30_avg - first_30_avg) / (first_30_avg + 1e-6)) * 100
    
    if growth_pct > 5:
        trend_label = f"an upward trajectory (+{growth_pct:.1f}%)"
    elif growth_pct < -5:
        trend_label = f"a downward trajectory ({growth_pct:.1f}%)"
    else:
        trend_label = "a steady, flat trajectory"

    # Extract Primary Drivers
    top_driver_name = "historical lags"
    if model is not None and hasattr(model, "feature_importances_") and feature_names:
        importances = model.feature_importances_
        top_idx = np.argmax(importances)
        top_driver_name = feature_names[top_idx]
    
    summary_data = {
        "total_volume": round(total_volume, 2),
        "avg_daily": round(avg_daily, 2),
        "peak_day": peak_day,
        "peak_val": round(peak_val, 2),
        "trend_label": trend_label,
        "top_driver": top_driver_name,
        "month": payload_dict.get("month", "N/A"),
        "quarter": payload_dict.get("quarter", "N/A")
    }

    # Backup template if Ollama is offline
    fallback_summary = (
        f"Over the next 90 days, projected revenue total is ${summary_data['total_volume']:,} "
        f"with a daily average of ${summary_data['avg_daily']:,} following {summary_data['trend_label']}. "
        f"Peak demand is expected on Day {summary_data['peak_day']} at ${summary_data['peak_val']:,}."
    )

    prompt = f"""
    You are an AI Executive Assistant in an enterprise analytics platform.
    Synthesize these model insights into a concise 2-sentence executive summary with a business recommendation:
    - 90-Day Forecast Volume: {summary_data['total_volume']}
    - Daily Average: {summary_data['avg_daily']}
    - Peak Demand: Day {summary_data['peak_day']} ({summary_data['peak_val']} units)
    - Trend: {summary_data['trend_label']}
    - Dominant Driver: {summary_data['top_driver']}
    - Context: Month {summary_data['month']}, Quarter {summary_data['quarter']}
    """

    # 2. Call Ollama Safely with Fallback
    try:
        response = ollama.generate(
            model="llama3",
            prompt=prompt
        )
        return response.get('response', fallback_summary).strip()
    except Exception as e:
        logging.warning(f"Ollama generation offline/failed: {e}. Utilizing fallback summary.")
        return fallback_summary
    
@app.route('/predict', methods=['POST'])
def predict():
    global GLOBAL_MODEL
    if GLOBAL_MODEL is None:
        logging.error("API Call failed: Model has not been trained yet.")
        return jsonify({"error": "Model is not trained or loaded yet."}), 500
        
    data = request.get_json(force=True)
    
    # features check
    required_keys = ['rolling_mn_1Mon', 'rolling_std_1Mon', 'lag_30', 'lag_60', 'lag_90', 'month', 'quarter']
    missing_keys = [key for key in required_keys if key not in data]
    
    if missing_keys:
        
        logging.warning(f"API Prediction requested with missing features: {missing_keys}")
        return jsonify({"error": f"Missing required features: {missing_keys}"}), 400
        
    # Convert incoming dictionary data into the correct shape for model input
    input_features = np.array([data[key] for key in required_keys]).reshape(1, -1)
    
    # Generate prediction array
    prediction = GLOBAL_MODEL.predict(input_features)[0]
    
        # 3. Generate Executive Summary
    summary = generate_executive_summary(
            predictions=prediction,
            payload_dict=data,
            model=GLOBAL_MODEL
        )
        
        # 4. Return combined response payload
        
    logging.info("Successfully generated 90-day forecast via API.")
    response_dict ={
        "status": "success",
        "forecast_horizon_days": 90,
        "predictions": prediction.tolist(),
        "executive_summary": summary,
        "metrics": {
                "total_projected_demand": round(sum(prediction), 2),
                "peak_day": int(np.argmax(prediction) + 1)
    }}
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"forecast_output_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(response_dict, f, indent=4)
        
    logging.info(f"Predictions saved locally to file: {filename}")
    

    return jsonify(response_dict)


            

if __name__ == "__main__":
    csv_path = 'C:\\Users\\hp\\AppData\\Local\\Temp\\all_training_data1.csv'
    if os.path.exists(csv_path):
        raw_data = ingest_data(csv_path)
        X, Y, daily_df = engineer_features(raw_data)
        GLOBAL_MODEL = train_production_model_RFR(raw_data,X, Y)
        
    app.run(host="0.0.0.0", port=5001, debug=False)


