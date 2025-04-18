import joblib
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

class LungCancerPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.load_model()

    def load_model(self):
        """Load the trained model"""
        try:
            model_path = r'C:\Users\BEXX\Desktop\Vericart3\Model\model.pkl'
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            self.model = joblib.load(model_path)
            print("\nModel loaded successfully")
            print("Model type:", type(self.model))
            return True
        except Exception as e:
            print(f"\nError loading model: {str(e)}")
            raise

    def preprocess_features(self, features):
        """Preprocess features to match training data"""
        processed = features.copy()
        
        # Convert gender: 1 (Male) -> 0, 2 (Female) -> 1
        processed[0] = 1 if processed[0] == 2 else 0
        
        # Scale age using the same range as training (assuming age 0-120)
        processed[1] = (processed[1] - 0) / (120 - 0)
        
        # Convert Yes/No values from 1/0 to match training
        for i in range(2, len(processed)):
            processed[i] = 1 if processed[i] == 1 else 0
            
        return processed

    def predict(self, symptoms):
        """Make a prediction"""
        try:
            # Print input symptoms for debugging
            print("\nOriginal input symptoms:")
            feature_names = ['Gender', 'Age', 'Smoking', 'Yellow Fingers', 'Anxiety', 
                           'Peer Pressure', 'Chronic Disease', 'Fatigue', 'Allergy',
                           'Wheezing', 'Alcohol Consuming', 'Coughing', 
                           'Shortness of Breath', 'Swallowing Difficulty', 'Chest Pain']
            for name, value in zip(feature_names, symptoms):
                print(f"{name}: {value}")
            
            # Validate number of features
            if len(symptoms) != 15:
                raise ValueError(f"Expected 15 features, got {len(symptoms)}")
            
            # Preprocess features
            processed_features = self.preprocess_features(symptoms)
            print("\nProcessed features:")
            for name, value in zip(feature_names, processed_features):
                print(f"{name}: {value}")
            
            # Reshape for prediction
            features = np.array(processed_features, dtype=float).reshape(1, -1)
            print("\nReshaped features:", features)
            print("Features shape:", features.shape)
            print("Features dtype:", features.dtype)
            
            # Make prediction
            prediction = self.model.predict(features)
            print("\nRaw prediction:", prediction)
            
            # Return binary prediction
            return int(prediction[0])
            
        except Exception as e:
            print(f"\nError during prediction: {str(e)}")
            raise

# Create a singleton instance
predictor = LungCancerPredictor()

def predict(symptoms):
    """Wrapper function for prediction"""
    return predictor.predict(symptoms)
