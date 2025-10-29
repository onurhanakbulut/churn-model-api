from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import os
import logging
from fastapi.middleware.cors import CORSMiddleware


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("churn_api")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
    )



TH_PATH = os.getenv('TH_PATH', 'app/threshold.txt')
MODEL_PATH = os.getenv('MODEL_PATH', 'app/churn_xgb_model.pkl')



model = None
threshold = 0.67


def load_threshold(path: str) -> float:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            val = float(f.read().strip())
        if not 0.0 <= val <= 1.0:
            raise ValueError(f"Threshold must be in [0,1], got {val}")
        LOGGER.info("Threshold loaded: %.4f (from %s)", val, path)
        return val
    except Exception as e:
        LOGGER.warning("Threshold load failed (%s). Using default 0.67", e)
        return 0.67




@app.on_event("startup")
def startup():
    global model, threshold
    model = joblib.load(MODEL_PATH)
    LOGGER.info("Model loaded from %s", MODEL_PATH)
    threshold = load_threshold(TH_PATH)


 
@app.get('/metadata')
def metadata():
    return {
        'model_path': MODEL_PATH,
        'threshold_path': TH_PATH,
        'threshold_value': threshold,
        'expected_features': ['Frequency', 'Monetary', 'CustomerLifetimeDays'],
        'model_type': type(model).__name__ if model is not None else None
    }




@app.get('/')
def read_root():
    return {'message': 'Churn Model API', 'threshold_used': threshold}

@app.get('/health')
def health():
    return {'status': 'ok' if model is not None else 'down'}

@app.get('/threshold')
def get_threshold():
    return {'threshold': threshold}

@app.post('/predict')
def predict(data: dict):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        features_list = [
            float(data['Frequency']),
            float(data['Monetary']),
            float(data['CustomerLifetimeDays']) 
            ]
        
        x = np.array(features_list, dtype=float).reshape(1, -1)
        
        y = model.predict(x)
        proba = float(model.predict_proba(x)[0][1])
        is_churn = bool(proba >= threshold)
        
        
        return {
            'predicted_class': int(y[0]),
            'churn_probability': proba,
            'is_churn': is_churn
            }
    
    except KeyError as ke:
        raise HTTPException(status_code=400, detail=f"Missing Field: {ke}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        
        
        
    
    










