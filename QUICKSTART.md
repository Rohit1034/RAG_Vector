"""
Quick Start Guide - Healthcare ML Pipeline
==========================================
Fast reference for running the ML pipeline
"""

# ============================================================
# INSTALLATION
# ============================================================

# Install dependencies
pip install -r requirements.txt

# ============================================================
# RUN COMPLETE PIPELINE (ALL STEPS)
# ============================================================

cd backend
python app/main.py --full

# This runs:
# 1. Data preprocessing
# 2. Feature engineering
# 3. Model training
# 4. Model evaluation

# ============================================================
# RUN INDIVIDUAL STEPS
# ============================================================

# 1. Data Preprocessing
python app/main.py --preprocess
# Or directly:
python training/preprocess.py

# 2. Feature Engineering
python app/main.py --feature
# Or directly:
python training/feature_engineering.py

# 3. Model Training
python app/main.py --train
# Or directly:
python training/train.py

# 4. Model Evaluation
python app/main.py --evaluate
# Or directly:
python training/evaluate.py

# 5. Example Prediction
python app/main.py --predict
# Or directly:
python app/predict.py

# ============================================================
# MAKING PREDICTIONS IN YOUR CODE
# ============================================================

from app.predict import predict_heart_disease

# Single patient prediction
patient_data = {
    'age': 63.0,
    'sex': 1,              # 1=male, 0=female
    'cp': 1,               # Chest pain type (1-4)
    'trestbps': 145.0,     # Resting BP
    'chol': 233.0,         # Cholesterol
    'fbs': 1,              # Fasting blood sugar
    'restecg': 2,          # Resting ECG
    'thalach': 150.0,      # Max heart rate
    'exang': 0,            # Exercise angina
    'oldpeak': 2.3,        # ST depression
    'slope': 3,            # ST slope
    'ca': 0.0,             # Number of vessels
    'thal': 6.0            # Thalassemia
}

result = predict_heart_disease(patient_data)

# Access results
print(f"Severity: {result['severity']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Risk: {result['risk_description']}")

# All class probabilities
for severity, prob in result['probabilities'].items():
    print(f"{severity}: {prob:.2%}")

# ============================================================
# BATCH PREDICTIONS
# ============================================================

from app.predict import batch_predict

# Predict for multiple patients from CSV
results_df = batch_predict('path/to/patients.csv')
results_df.to_csv('predictions.csv', index=False)

# ============================================================
# VIEW RESULTS
# ============================================================

# Model artifacts location:
# - Trained model: model/trained_model.pkl
# - Scaler: model/scaler.pkl
# - Feature importance: model/feature_importance.pkl

# Visualizations:
# - Confusion matrix: model/confusion_matrix.png
# - Feature importance: model/feature_importance.png
# - Correlation matrix: model/correlation_matrix.png

# ============================================================
# TROUBLESHOOTING
# ============================================================

# If you get "Model not found" error:
# Run training first:
python training/train.py

# If you get "Data not found" error:
# Run preprocessing first:
python training/preprocess.py

# If you get feature mismatch error:
# Make sure your patient data has all 13 required features:
# age, sex, cp, trestbps, chol, fbs, restecg, 
# thalach, exang, oldpeak, slope, ca, thal

# ============================================================
# PROJECT STRUCTURE
# ============================================================

"""
healthcare-ml-project/
├── backend/
│   ├── app/
│   │   ├── main.py          ← Main entry point
│   │   ├── predict.py       ← Prediction API
│   │   └── config.py        ← Configuration
│   ├── training/
│   │   ├── preprocess.py    ← Step 1
│   │   ├── feature_engineering.py  ← Step 2
│   │   ├── train.py         ← Step 3
│   │   ├── evaluate.py      ← Step 4
│   │   └── utils.py         ← Helpers
│   └── requirements.txt
├── model/                   ← Saved models & plots
├── data/
│   ├── raw/                 ← Original data
│   └── processed/           ← Processed data
└── README.md
"""
