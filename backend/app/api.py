from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib

from app.rag.rag_pipeline import load_knowledge, retrieve_context
from app.rag.llm_explainer import generate_explanation

from app.config import (
    TRAINED_MODEL_FILE,
    SCALER_FILE,
    FEATURE_COLUMNS_FILE,
    CLASS_LABELS
)

# ============================================================
# INIT
# ============================================================

app = FastAPI(title="Healthcare ML API")

print("🔄 Loading model...")

model = joblib.load(TRAINED_MODEL_FILE)
scaler = joblib.load(SCALER_FILE)
feature_columns = joblib.load(FEATURE_COLUMNS_FILE)

db = load_knowledge()

print("✅ Model loaded successfully")
print("📊 Expected features:", feature_columns)

# ============================================================
# INPUT SCHEMA
# ============================================================

class PatientData(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

# ============================================================
# ROOT
# ============================================================

@app.get("/")
def home():
    return {"message": "API running 🚀"}

# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
def predict(data: PatientData):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([data.model_dump()])

        print("\n📥 Incoming input:", input_df.columns.tolist())

        # 🔥 FIX: Ensure all required features exist
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Keep only required features in correct order
        input_df = input_df[feature_columns]

        print("📊 Final input columns:", input_df.columns.tolist())

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Prediction
        prediction = model.predict(input_scaled)[0]
        severity = CLASS_LABELS.get(int(prediction), "Unknown")

        # RAG context
        query = f"Patient with age {data.age}, cholesterol {data.chol}, chest pain {data.cp}"
        context = retrieve_context(db, query)

        # LLM explanation
        explanation = generate_explanation(
            prediction,
            severity,
            context,
            data.model_dump()
        )

        return {
            "prediction": int(prediction),
            "severity": severity,
            "explanation": explanation
        }

    except Exception as e:
        print("❌ ERROR:", str(e))
        return {"error": str(e)}