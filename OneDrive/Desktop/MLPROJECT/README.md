# Aviation Fleet Predictive Maintenance Platform

##  Overview
This project is an end-to-end predictive maintenance platform for aircraft engine fleets. It processes engine telemetry, trains machine learning models to predict Remaining Useful Life (RUL), and provides interactive dashboards for fleet risk analysis.

The platform is inspired by aerospace analytics practices and is designed for real-world aircraft health monitoring and decision support.

---

##  Key Components

###  Data Processing
- Loads NASA CMAPSS run-to-failure aircraft engine data
- Computes Remaining Useful Life (RUL)
- Stores processed data for analytics and modeling

###  ML Model
- Random Forest Regression model to predict RUL
- Evaluation metrics (RMSE, R², bias analysis)
- Saved artifacts (model, scaler, feature names, metrics)

###  REST API
- FastAPI service to serve predictions on demand
- Enables integration with dashboards or external systems

###  Dashboard
- Streamlit + Plotly interactive interface
- Fleet overview, RUL distributions, engine drill-down
- Business metrics and risk prioritization

---

##  Tech Stack

| Technology | Use Case |
|------------|----------|
| Python | Core language |
| Pandas / NumPy | Data processing |
| Scikit-learn | ML modeling |
| FastAPI | Prediction API |
| Streamlit | Interactive dashboards |
| Plotly | Visualizations |

---

##  Installation & Setup

### 1. Clone the repo

```bash
git clone https://github.com/bharadwajpp/aviation-fleet-predictive-maintenance.git
cd aviation-fleet-predictive-maintenance
2. Create virtual environment
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate # macOS / Linux

3. Install dependencies
pip install -r requirements.txt

4. Run the API
uvicorn api.predict_api:app --reload

5. Run the Dashboard
streamlit run app/dashboard.py

Project Structure
data/            # Raw & processed data
src/             # Core data processing & ML
app/             # Streamlit dashboard
api/             # FastAPI prediction API
models/          # Saved model artifacts
outputs/         # CSV/metrics/export

Features

RUL prediction for each engine cycle

Fleet risk scoring

Interactive dashboards

API for real-time predictions