🧠 Churn Model API (XGBoost · FastAPI)

**https://churn-dashboard-demo.streamlit.app**

A lightweight and production-ready REST API that serves an e-commerce churn prediction model trained with XGBoost.
It takes customer behavioral features and returns their churn probability — helping businesses identify at-risk customers.

🚀 Features

⚡ Fast inference powered by FastAPI + Uvicorn

✅ Strict request validation via Pydantic

🧩 Model path configurable with environment variable (MODEL_PATH)

🐳 Containerized with Docker

🌍 CORS-enabled — easily integrates with Streamlit dashboards

📦 Tech Stack

Python • FastAPI • XGBoost • scikit-learn • joblib • Pydantic • Docker

🔌 API Overview
Health Check
GET /health
→ 200 OK
{
  "status": "ok"
}

Predict (Single Record)
POST /predict
Content-Type: application/json
Body:
{
  "Frequency": 12,
  "Monetary": 358.4,
  "CustomerLifetimeDays": 210
}
Response (200):
{
  'predicted_class': 1,
  'churn_probability': 0.78,
  'is_churn': Churned
}
---------

⚙️ Quickstart
1️⃣ Clone the repository
git clone https://github.com/onurhanakbulut/churn-model-api.git
cd churn-model-api

2️⃣ Install dependencies
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

3️⃣ Add your trained model

Place your serialized model (e.g. joblib.dump) inside:

app/churn_xgb_model.pkl


Or specify a custom model path:

set MODEL_PATH=app/churn_xgb_model.pkl  # Windows
export MODEL_PATH=app/churn_xgb_model.pkl  # Linux/macOS

4️⃣ Run locally
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload


Then visit:
👉 http://localhost:8000/docs

🐳 Docker Usage
Build image
docker build -t churn-model-api:latest .

Run container
docker run --rm -p 8000:8000 \
  -e MODEL_PATH=/app/app/churn_xgb_model.pkl \
  churn-model-api:latest

⚙️ Environment Variables
Variable	Default	Description
MODEL_PATH	app/churn_xgb_model.pkl	Path to serialized .pkl model file
HOST	0.0.0.0	Bind host for Uvicorn
PORT	8000	Server port
LOG_LEVEL	info	Log level: critical/error/warning/info/debug
🧪 Example Requests

Using cURL

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Frequency": 12, "Monetary": 358.4, "CustomerLifetimeDays": 210}'


Using Python

import requests

payload = {"Frequency": 12, "Monetary": 358.4, "CustomerLifetimeDays": 210}
r = requests.post("http://localhost:8000/predict", json=payload)
print(r.json())  # → {'churn_probability': 0.27}


👩‍💻 Author

Onurhan Akbulut
Data Science & Machine Learning Engineer
📬 onurhanakbulut
