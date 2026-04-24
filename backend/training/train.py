# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import sys

sys.path.append(str(Path(__file__).parent.parent))
from app.config import *

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# LOAD DATA (🔥 FIXED HERE)
# ============================================================

def load_processed_data(file_path=None):
    if file_path is None:
        file_path = PROCESSED_DATA_DIR / "final_processed_data.csv"

    logger.info(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)

    # Separate
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # 🔥 CRITICAL FIX
    y = np.round(y).astype(int)
    y = y - y.min()

    logger.info(f"Fixed target classes: {np.unique(y)}")

    return X, y

# ============================================================
# SPLIT
# ============================================================

def split_data(X, y):
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(X_train, y_train):

    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        objective='multi:softmax',
        num_class=len(np.unique(y_train))
    )

    logger.info("Training model...")
    model.fit(X_train, y_train)

    return model

# ============================================================
# SAVE MODEL
# ============================================================

def save_model(model):
    joblib.dump(model, TRAINED_MODEL_FILE)
    logger.info(f"Model saved at: {TRAINED_MODEL_FILE}")

# ============================================================
# MAIN PIPELINE
# ============================================================

def training_pipeline():
    logger.info("🚀 STARTING TRAINING")

    X, y = load_processed_data()

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)

    save_model(model)

    logger.info("✅ TRAINING COMPLETE")

    return model

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    training_pipeline()