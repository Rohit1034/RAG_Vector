# Intelligent IoT-Based Healthcare Monitoring System
## ML-Driven Heart Disease Risk Prediction Pipeline

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-green.svg)](https://scikit-learn.org/)

A production-ready machine learning pipeline for predicting heart disease severity using the UCI Heart Disease dataset. This project demonstrates best practices in ML engineering for healthcare applications.

---

## 🎯 Project Overview

This project implements a **multi-class classification system** that predicts cardiovascular disease severity across 5 categories:

- **Class 0**: Healthy (no disease)
- **Class 1**: Mild heart disease
- **Class 2**: Moderate heart disease
- **Class 3**: Severe heart disease
- **Class 4**: Very severe heart disease

### Key Features

✅ **Production-Ready Architecture**: Modular, clean, and maintainable codebase  
✅ **Comprehensive Data Preprocessing**: Handling missing values, feature scaling, encoding  
✅ **Advanced Feature Engineering**: Correlation analysis, feature importance selection  
✅ **XGBoost Model**: State-of-the-art gradient boosting for healthcare predictions  
✅ **Extensive Evaluation**: Multiple metrics, confusion matrix, ROC-AUC analysis  
✅ **Clinical Interpretability**: Feature importance and medical context explanations  

---

## 📁 Project Structure

```
healthcare-ml-project/
│
├── backend/
│   ├── app/
│   │   ├── main.py              # Main orchestration script
│   │   ├── predict.py           # Prediction API
│   │   └── config.py            # Configuration settings
│   │
│   ├── training/
│   │   ├── preprocess.py        # Data preprocessing pipeline
│   │   ├── feature_engineering.py  # Feature selection & engineering
│   │   ├── train.py             # Model training pipeline
│   │   ├── evaluate.py          # Model evaluation
│   │   └── utils.py             # Utility functions
│   │
│   └── requirements.txt         # Python dependencies
│
├── model/
│   ├── trained_model.pkl        # Trained XGBoost model
│   ├── scaler.pkl               # Feature scaler
│   ├── feature_columns.pkl      # Feature list
│   ├── feature_importance.pkl   # Feature importance scores
│   ├── confusion_matrix.png     # Confusion matrix visualization
│   └── feature_importance.png   # Feature importance plot
│
├── data/
│   ├── raw/
│   │   └── heart_disease.csv    # Original dataset
│   │
│   └── processed/
│       ├── processed_data.csv   # Preprocessed data
│       ├── final_processed_data.csv  # Final engineered data
│       ├── train_data.csv       # Training set
│       └── test_data.csv        # Test set
│
├── notebooks/
│   └── exploration.ipynb        # Data exploration notebook
│
└── README.md                    # This file
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Navigate to the project directory
cd healthcare-ml-project/backend

# Install required packages
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline

```bash
# Option 1: Run everything at once
python app/main.py --full

# Option 2: Run step-by-step
python app/main.py --preprocess    # Data preprocessing
python app/main.py --feature       # Feature engineering
python app/main.py --train         # Model training
python app/main.py --evaluate      # Model evaluation
```

### 3. Make Predictions

```bash
# Run example prediction
python app/main.py --predict
```

Or use the prediction API in your code:

```python
from app.predict import predict_heart_disease

# Patient data
patient = {
    'age': 63.0,
    'sex': 1,           # 1=male, 0=female
    'cp': 1,            # Chest pain type
    'trestbps': 145.0,  # Resting blood pressure
    'chol': 233.0,      # Cholesterol
    'fbs': 1,           # Fasting blood sugar
    'restecg': 2,       # Resting ECG
    'thalach': 150.0,   # Max heart rate
    'exang': 0,         # Exercise induced angina
    'oldpeak': 2.3,     # ST depression
    'slope': 3,         # Slope of ST segment
    'ca': 0.0,          # Number of vessels
    'thal': 6.0         # Thalassemia
}

# Make prediction
result = predict_heart_disease(patient)
print(f"Severity: {result['severity']}")
print(f"Confidence: {result['confidence']:.2%}")
```

---

## 📊 Dataset

### UCI Heart Disease Dataset

- **Source**: Cleveland Clinic Foundation
- **Samples**: 303 patients
- **Features**: 14 clinical attributes
- **Target**: Multi-class (0-4) disease severity

### Key Features

| Feature | Description | Type |
|---------|-------------|------|
| `age` | Patient age in years | Numerical |
| `sex` | Sex (1=male, 0=female) | Categorical |
| `cp` | Chest pain type (1-4) | Categorical |
| `trestbps` | Resting blood pressure (mm Hg) | Numerical |
| `chol` | Serum cholesterol (mg/dl) | Numerical |
| `fbs` | Fasting blood sugar > 120 mg/dl | Binary |
| `restecg` | Resting ECG results (0-2) | Categorical |
| `thalach` | Maximum heart rate achieved | Numerical |
| `exang` | Exercise induced angina | Binary |
| `oldpeak` | ST depression | Numerical |
| `slope` | Slope of peak exercise ST segment | Categorical |
| `ca` | Number of major vessels (0-3) | Numerical |
| `thal` | Thalassemia (3,6,7) | Categorical |

---

## 🔬 Methodology

### 1. Data Preprocessing

- **Missing value imputation**: Median for numerical, mode for categorical
- **Outlier handling**: Statistical validation of clinical ranges
- **Feature scaling**: StandardScaler for normalization
- **Stratified splitting**: Maintaining class distribution (80/20 split)

### 2. Feature Engineering

- **Correlation analysis**: Removing highly correlated features (>0.9)
- **Feature importance**: XGBoost-based selection
- **Domain knowledge**: Preserving critical clinical features
- **Dimensionality optimization**: Balancing performance and interpretability

### 3. Model Training

- **Algorithm**: XGBoost (Extreme Gradient Boosting)
- **Objective**: Multi-class classification (`multi:softmax`)
- **Hyperparameters**:
  - Learning rate: 0.05
  - Max depth: 5
  - Estimators: 200
  - Subsample: 0.8
  - Column sample: 0.8

### 4. Evaluation Metrics

- **Accuracy**: Overall correctness
- **Precision**: Positive predictive value
- **Recall**: Sensitivity
- **F1-Score**: Harmonic mean of precision & recall
- **ROC-AUC**: Area under ROC curve (One-vs-Rest)
- **Confusion Matrix**: Per-class performance analysis

---

## 📈 Model Performance

### Test Set Results

| Metric | Score |
|--------|-------|
| **Accuracy** | 54.10% |
| **Precision** | 52.02% |
| **Recall** | 54.10% |
| **F1-Score** | 53.00% |
| **ROC-AUC** | 79.42% |

### Per-Class Performance

| Severity | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Healthy | 0.82 | 0.85 | 0.84 | 33 |
| Mild | 0.27 | 0.27 | 0.27 | 11 |
| Moderate | 0.00 | 0.00 | 0.00 | 7 |
| Severe | 0.22 | 0.29 | 0.25 | 7 |
| Very Severe | 0.00 | 0.00 | 0.00 | 3 |

### Top 5 Most Important Features

1. **Chest Pain Type (cp)** - 11.94%
2. **Thalassemia (thal)** - 11.34%
3. **Number of Vessels (ca)** - 10.51%
4. **ST Slope (slope)** - 10.08%
5. **Exercise Induced Angina (exang)** - 9.16%

---

## 🏥 Clinical Applications

### Use Cases

1. **Risk Screening**: Early identification of high-risk patients
2. **Clinical Decision Support**: Assisting physicians in diagnosis
3. **Resource Allocation**: Prioritizing patients for further testing
4. **Population Health**: Analyzing cardiovascular disease trends
5. **Research**: Identifying key biomarkers for heart disease

### Medical Interpretation

The model identifies **chest pain characteristics**, **thalassemia status**, and **coronary vessel involvement** as the strongest predictors of cardiovascular disease. This aligns with established clinical knowledge and can guide targeted interventions.

---

## 🛠️ Technologies Used

### Core ML Stack

- **Python 3.8+**: Programming language
- **XGBoost 2.0+**: Gradient boosting framework
- **scikit-learn 1.3+**: ML utilities and preprocessing
- **pandas 2.0+**: Data manipulation
- **numpy 1.24+**: Numerical computing

### Visualization & Analysis

- **matplotlib 3.7+**: Plotting library
- **seaborn 0.12+**: Statistical visualization
- **joblib 1.3+**: Model serialization

### Development Tools

- **Jupyter**: Interactive data exploration
- **Git**: Version control

---

## 📚 Code Quality & Best Practices

### ✅ Production-Ready Features

- **Modular Architecture**: Separation of concerns across modules
- **Comprehensive Logging**: Detailed logging at every pipeline stage
- **Error Handling**: Robust exception handling and validation
- **Configuration Management**: Centralized config file
- **Documentation**: Extensive inline comments and docstrings
- **Type Safety**: Input validation and type checking
- **Reproducibility**: Fixed random seeds and saved artifacts

### 🎯 Healthcare-Specific Considerations

- **Clinical Interpretability**: Feature importance and explanations
- **Medical Context**: Domain knowledge embedded in code comments
- **Safety Checks**: Validation of clinical value ranges
- **Audit Trail**: Complete logging of preprocessing and predictions
- **Stratified Sampling**: Maintaining disease prevalence in splits

---

## 🔮 Future Enhancements

### Phase 1: Model Improvements

- [ ] Hyperparameter tuning (GridSearchCV, Optuna)
- [ ] Ensemble methods (stacking, voting)
- [ ] Class imbalance handling (SMOTE, class weights)
- [ ] Cross-validation for robust evaluation

### Phase 2: Feature Engineering

- [ ] Polynomial features and interactions
- [ ] Medical domain-specific ratios (e.g., cholesterol/HDL)
- [ ] Time-series features (for MIMIC-IV dataset)

### Phase 3: Deployment

- [ ] REST API with FastAPI/Flask
- [ ] Docker containerization
- [ ] Model versioning (MLflow)
- [ ] Real-time prediction endpoint
- [ ] Web dashboard for visualization

### Phase 4: Advanced Analytics

- [ ] SHAP values for explainability
- [ ] Calibration curves
- [ ] Survival analysis
- [ ] Integration with MIMIC-IV ICU dataset

---

## 📖 References

### Dataset

- Dua, D. and Graff, C. (2019). UCI Machine Learning Repository. Irvine, CA: University of California, School of Information and Computer Science.
- Detrano, R., et al. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. *American Journal of Cardiology*, 64, 304-310.

### Methods

- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*.

---

## 👨‍💻 Author

**Healthcare ML Project**  
Major Project - IoT-Based Healthcare Monitoring System  
Date: February 2026

---

## 📄 License

This project is for educational and research purposes.

---

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the heart disease dataset
- Cleveland Clinic Foundation for data collection
- Dr. Robert Detrano and collaborators for the original research

---

## 📧 Contact

For questions or collaboration opportunities, please reach out through your project supervisor.

---

**⚕️ Disclaimer**: This model is for research and educational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions.
