import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt  
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 

# Load dataset
master_df = pd.read_csv('C:\\Users\\hp\\AppData\\Local\\Temp\\all_training_data1.csv', low_memory=False)

# Clean out double column conventions safely
master_df['price'] = master_df['price'].fillna(master_df['total_price'])
master_df['stream_id'] = master_df['stream_id'].fillna(master_df['StreamID'])
master_df['times_viewed'] = master_df['times_viewed'].fillna(master_df['TimesViewed'])
master_df.drop(columns=['total_price', 'StreamID', 'TimesViewed'], inplace=True, errors='ignore')


master_df['date'] = pd.to_datetime(master_df[['year', 'month', 'day']])

print("--- 1. Aggregating Daily Distribution Details ---")

daily_df = master_df.groupby('date').agg(
    total_value=('price', 'sum'),
    transaction_count=('price', 'count'),
    country_count=('country', 'nunique'),
    invoices_count=('invoice', 'count'),
    times_viewed=('times_viewed', 'sum')
).reset_index().sort_values(by='date', ascending=True)

print(daily_df.head(10))
daily_df.to_csv("total_value_by_day.csv", index=False)

print("_______________________________________________________")
print("Country Performance Analysis:")
total_revenue_by_Country = master_df.groupby('country').agg(
    total_value=('price', 'sum'),
    transaction_count=('price', 'count'),
    invoices_count=('invoice', 'count')
).reset_index().sort_values(by='total_value', ascending=False) 

print(total_revenue_by_Country.head(10))
total_revenue_by_Country.to_csv("total_value_by_country.csv", index=False)

print("_______________________________________________________")
print("Purchases Summary By Months:")
months_df = master_df.groupby('month').agg(
    total_revenue=('price', 'sum'),            
    transaction_count=('invoice', 'count'),
    countries_count=('country', 'nunique')
).reset_index().sort_values('month')
print(months_df)


min_date, max_date = master_df['date'].min(), master_df['date'].max()
date_span = (max_date - min_date).days + 1
print(f"\nEarliest Date: {min_date.date()} | Latest Date: {max_date.date()} | Total Range: {date_span} days")


# ================= Visualization =====================

master_df['month_year'] = master_df['date'].dt.to_period('M')

revenue_by_monthes = master_df.groupby('month_year').agg(
    total_revenue=('price', 'sum'),            
    transaction_count=('invoice', 'count')
).reset_index().sort_values('month_year')

revenue_by_monthes['month_year_str'] = revenue_by_monthes['month_year'].astype(str)
revenue_by_monthes['revenue_by_3monthes'] = revenue_by_monthes['total_revenue'].rolling(window=3, min_periods=1).mean()

plt.figure(figsize=(12, 6))
plt.plot(revenue_by_monthes['month_year_str'], revenue_by_monthes['total_revenue'], marker='o', label="Monthly Revenue")
plt.plot(revenue_by_monthes['month_year_str'], revenue_by_monthes['revenue_by_3monthes'], label="3-Month Moving Average", linewidth=2.5, linestyle='--')
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.title("Monthly Revenue with 3-Month Moving Average")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# =====================================================
# ==================== Modeling =======================

print("\n--- 2. Building Features for Machine Learning ---")

# Apply rolling computations safely directly onto the daily data timeline
daily_df['rolling_mn_1Mon'] = daily_df['total_value'].rolling(window=30, min_periods=1).mean()
daily_df['rolling_std_1Mon'] = daily_df['total_value'].rolling(window=30, min_periods=1).std()

for lag in [30, 60, 90]:
    daily_df[f'lag_{lag}'] = daily_df['total_value'].shift(lag)
    
daily_df['month'] = daily_df['date'].dt.month
daily_df['quarter'] = daily_df['date'].dt.quarter

# Generate multipoint forecasting matrices (90-day targets)
forecast_horizon = 90
target_columns = []
for step in range(1, forecast_horizon + 1):
    col_name = f'target_day_{step}'
    daily_df[col_name] = daily_df['total_value'].shift(-step)
    target_columns.append(col_name)

# Drop rows with NaNs caused by lag features and target shifts
daily_df = daily_df.dropna()

feature_cols = ['rolling_mn_1Mon', 'rolling_std_1Mon', 'lag_30', 'lag_60', 'lag_90', 'month', 'quarter']
X = daily_df[feature_cols]
Y = daily_df[target_columns]

# --- Naive Baseline Evaluation  ---
revenue_by_monthes['baseline_pred'] = revenue_by_monthes['total_revenue'].shift(1)
clean_baseline = revenue_by_monthes[['total_revenue', 'baseline_pred']].dropna()

baseline_mae = mean_absolute_error(clean_baseline['total_revenue'], clean_baseline['baseline_pred'])
print(f"Monthly Baseline Naive MAE: {baseline_mae:.2f}")

#  TimeSeries Cross Validation 
tscv = TimeSeriesSplit(n_splits=10)
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
scores = []

print("\nRunning Cross-Validation Window Splits...")
for fold, (train_index, test_index) in enumerate(tscv.split(X), 1):
    x_train, x_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
    
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    mae = mean_absolute_error(y_test, preds)
    scores.append(mae)
    print(f" Fold {fold} MAE: {mae:.2f}")
    
print("-------------------------------------------------------")
print(f"All Fold Target MAEs: {[round(s, 2) for s in scores]}")
print(f"Average Cross-Validated MAE: {np.mean(scores):.2f}")