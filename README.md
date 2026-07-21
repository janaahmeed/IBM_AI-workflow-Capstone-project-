# IBM_AI-workflow-Capstone-project-
An enterprise 90-day revenue forecasting engine using Random Forest regression, a Flask REST API backend, an interactive Streamlit UI, and local LLM executive summaries via Ollama.

* **Frontend:** Streamlit[cite: 2], Pandas
* **Backend API:** Flask[cite: 1], Requests[cite: 2]
* **Machine Learning:** Scikit-Learn (`RandomForestRegressor`[cite: 1]  , `TimeSeriesSplit`[cite: 1]), NumPy
* **AI Synthesis:** Ollama (`llama3`) with automated fallback strings[cite: 1]

---

## 📂 Project Structure

```text
├── app_core_logging.py    # Flask REST API, model training pipeline, feature engineering, and LLM synthesis
├── app_ui.py              # Streamlit user interface and interactive data visualization
├── requirements.txt       # Project dependencies
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


