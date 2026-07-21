import streamlit as st
import requests
import pandas as pd

# Page Configuration
st.set_page_config(page_title="Aavail Revenue Forecast Dashboard", layout="wide")

st.title("90-Day Revenue Forecasting Engine")
st.markdown("Interact with the Random Forest multi-point forecasting model and AI Executive Summary.")

col1, col2 = st.columns([1, 2])
with col1:
    st.header("Input Lag Features")
    st.caption("Provide recent metrics to seed the forecast horizon.")
    
    rolling_mn = st.number_input("30-Day Rolling Revenue Mean ($)", min_value=0.0, value=2500.0, step=50.0)
    rolling_std = st.number_input("30-Day Rolling Revenue Std Dev ($)", min_value=0.0, value=350.0, step=10.0)
    
    lag_30 = st.number_input("Revenue 30 Days Ago ($)", min_value=0.0, value=2400.0, step=50.0)
    lag_60 = st.number_input("Revenue 60 Days Ago ($)", min_value=0.0, value=2600.0, step=50.0)
    lag_90 = st.number_input("Revenue 90 Days Ago ($)", min_value=0.0, value=2300.0, step=50.0)
    
    target_month = st.slider("Target Execution Month", min_value=1, max_value=12, value=7)
    target_quarter = (target_month - 1) // 3 + 1
    st.info(f"Automatically inferred Quarter: {target_quarter}")

    submit_btn = st.button("Generate Forecast", type="primary")

with col2:
    st.header(" Multipoint Horizon Projection")
    
    if submit_btn:
        payload = {
            "rolling_mn_1Mon": rolling_mn,
            "rolling_std_1Mon": rolling_std,
            "lag_30": lag_30,
            "lag_60": lag_60,
            "lag_90": lag_90,
            "month": target_month,
            "quarter": target_quarter
        }
        
        FLASK_API_URL = "http://localhost:5001/predict"
        
        try:
            with st.spinner("Fetching forecast & AI Executive Summary..."):
                # Make the actual API call to Flask
                response = requests.post(FLASK_API_URL, json=payload, timeout=15)
            
            if response.status_code == 200:
                try:
                    res_data = response.json()
                    predictions = res_data.get('predictions', [])
                    summary = res_data.get('executive_summary', '')
                    metrics = res_data.get('metrics', {})
                    
                    # 1. AI Executive Summary
                    st.subheader("💡 AI Executive Summary")
                    st.info(summary)
                    
                    # 2. Key Metrics Cards
                    m_col1, m_col2 = st.columns(2)
                    m_col1.metric("90-Day Total Demand", f"${metrics.get('total_projected_demand', 0):,.2f}")
                    m_col2.metric("Peak Demand Day", f"Day {metrics.get('peak_day', 'N/A')}")
                    
                    # 3. 90-Day Time-Series Chart
                    forecast_days = [f"Day {i}" for i in range(1, len(predictions) + 1)]
                    chart_df = pd.DataFrame({
                        "Timeline Horizon": forecast_days,
                        "Predicted Daily Revenue ($)": predictions
                    }).set_index("Timeline Horizon")
                    
                    st.line_chart(chart_df, height=350)
                    
                    with st.expander("🔍 View Raw Prediction Vector Array"):
                        st.dataframe(chart_df.T)

                except requests.exceptions.JSONDecodeError:
                    st.error("API returned 200 OK, but response body was not valid JSON.")

            else:
                # Safely parse JSON error, or fall back to raw HTML/text output
                try:
                    err_msg = response.json().get('error', response.text)
                except requests.exceptions.JSONDecodeError:
                    err_msg = response.text or f"HTTP {response.status_code} Error"
                    
                st.error(f"API Error ({response.status_code}): {err_msg}")

        except requests.exceptions.ConnectionError:
            st.error("🔌 Connection Refused: Could not reach Flask server on http://localhost:5001!")
            