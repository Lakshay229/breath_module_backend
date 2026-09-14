import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ------------------------------------------------------------------
# App setup
# ------------------------------------------------------------------
app = FastAPI(
    title="HCC Biomarker Prediction API",
    description="Predicts liver tumor risk from ammonia, hydrogen sulphide, acetone, methane, and temperature sensor readings.",
    version="1.0.0",
)

# Allow the Flutter app (and anything else) to call this API from any origin.
# Tighten allow_origins to your app's actual domain(s) once you're past testing.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# ------------------------------------------------------------------
# Load models + scaler at startup
# ------------------------------------------------------------------
models = {}
scaler = None

@app.on_event("startup")
def load_artifacts():
    global scaler
    try:
        models["rf"] = joblib.load(os.path.join(MODEL_DIR, "model_random_forest.joblib"))
        models["xgb"] = joblib.load(os.path.join(MODEL_DIR, "model_xgboost.joblib"))
        models["knn"] = joblib.load(os.path.join(MODEL_DIR, "model_knn.joblib"))
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
        print(f"Loaded models: {list(models.keys())}")
    except Exception as e:
        print(f"WARNING: could not load one or more artifacts — {e}")


# ------------------------------------------------------------------
# Request / response schemas
# ------------------------------------------------------------------
class PredictionRequest(BaseModel):
    ammonia: float = Field(..., description="Ammonia sensor reading")
    hydrogen_sulphide: float = Field(..., description="Hydrogen sulphide sensor reading")
    acetone: float = Field(..., description="Acetone sensor reading")
    methane: float = Field(..., description="Methane sensor reading")
    temperature: float = Field(..., description="Temperature reading (°C)")
    model: str = Field("rf", description="Which model to use: 'rf', 'xgb', or 'knn'")


class PredictionResponse(BaseModel):
    model_used: str
    prediction: int
    label: str
    probability: float


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "HCC Biomarker Prediction API is running.",
        "available_models": list(models.keys()),
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if models else "models not loaded",
        "models_loaded": list(models.keys()),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    if req.model not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{req.model}'. Choose one of: {list(models.keys())}",
        )
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded on server.")

    # Feature order MUST match training order: acetone, ammonia, hydrogen_sulfide, methane, temperature
    features = np.array([[req.acetone, req.ammonia, req.hydrogen_sulphide, req.methane, req.temperature]])

    chosen_model = models[req.model]

    # KNN was trained on scaled features — scale for it, leave RF/XGB raw
    if req.model == "knn":
        features_input = scaler.transform(features)
    else:
        features_input = features

    pred = int(chosen_model.predict(features_input)[0])
    prob = float(chosen_model.predict_proba(features_input)[0][1])

    return PredictionResponse(
        model_used=req.model,
        prediction=pred,
        label="positive" if pred == 1 else "negative",
        probability=round(prob, 4),
    )