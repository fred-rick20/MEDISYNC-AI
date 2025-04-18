import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.svm import SVC
from .Breast_cancer_prediction_final import predict as predict_internal

def load_model():
    """Load the trained breast cancer prediction model."""
    try:
        model = joblib.load('Model/breast_cancer/breast_cancer_model.joblib')
        scaler = joblib.load('Model/breast_cancer/scaler.joblib')
        return model, scaler
    except:
        print("Error loading model files")
        return None, None

def predict(features, algorithm='svm'):
    """
    Wrapper function for breast cancer prediction.
    Args:
        features: List of numerical features for prediction
        algorithm: String indicating which algorithm to use ('svm', 'random_forest', or 'logistic')
    Returns:
        Dictionary containing prediction results
    """
    return predict_internal(features, algorithm) 