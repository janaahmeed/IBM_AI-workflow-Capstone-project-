# IBM_AI-workflow-Capstone-project-90-day revenue forecasting engine
An enterprise 90-day revenue forecasting engine using Random Forest regression, a Flask REST API backend, an interactive Streamlit UI, and local LLM executive summaries via Ollama.

* **Frontend:** Streamlit[cite: 2], Pandas
* **Backend API:** Flask[cite: 1], Requests[cite: 2]
* **Unit testing & logging file for mintoring Model performance  **
* ** Data fetching ,EDA , feature engineering & lagged features **
* **Machine Learning:** Scikit-Learn (`RandomForestRegressor`[cite: 1]  , `TimeSeriesSplit`[cite: 1]), NumPy
* **AI Synthesis:** Ollama (`llama3`) with automated fallback strings[cite: 1]

---

## 📂 Project Structure

```text
├── app_core_logging.py    # Flask REST API, model training pipeline, feature engineering, and LLM synthesis
├── app_ui.py              # Streamlit user interface and interactive data visualization
├── test_suite .py 
└── README.md              # Project documentation
#machine-learning #time-series-forecasting #random-forest #flask #streamlit #ollama #python #data-science
## API Endpoint Reference

{
  "rolling_mn_1Mon": 2500.0,
  "rolling_std_1Mon": 350.0,
  "lag_30": 2400.0,
  "lag_60": 2600.0,
  "lag_90": 2300.0,
  "month": 7,
  "quarter": 3
}


# End-to-End Automated Pipeline: Automates data ingestion, cleaning, and time-series aggregation to output structured daily and country-level business analytics.
# Multi-Output Forecast Engine: Employs a multi-output RandomForestRegressor to project a continuous 90-day revenue vector simultaneously.
#Mathematical Safeguards: Implements target log-transformation log1p , expm1 to stabilize high variance in target variables, paired with zero-clipping post-processing max(0, y)to ensure no negative revenue predictions hit production.
#Robust Temporal Validation: Evaluates model stability across 20 temporal cross-validation folds using TimeSeriesSplit, auto-generating audit records in local JSON logs.
#Generative AI Executive Synthesis: Integrates an offline-resilient local LLM pipeline (ollama / llama3) that converts prediction arrays into multi-metric executive summaries with natural language business recommendations.
#Interactive Decoupled Architecture: Features a decoupled Flask REST API backend paired with a dynamic Streamlit frontend, supporting scenario modeling across rolling windows and seasonal execution months.  
