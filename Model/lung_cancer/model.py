import numpy as np
import joblib
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

def predict(features, algorithm='svm'):
    """
    Make a lung cancer prediction using the specified algorithm.
    Args:
        features: List of numerical features for prediction
        algorithm: String indicating which algorithm to use ('svm', 'knn', or 'gradient_boosting')
    Returns:
        Dictionary containing prediction results
    """
    try:
        # Determine model path based on algorithm
        if algorithm == 'svm':
            model_path = os.path.join(current_dir, 'model.pkl')
        elif algorithm == 'knn':
            model_path = os.path.join(current_dir, 'knn_model.pkl')
        elif algorithm == 'gradient_boosting':
            model_path = os.path.join(current_dir, 'gradient_boosting_model.pkl')
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Load the model
        model = joblib.load(model_path)
        
        # Convert features to numpy array and reshape
        features = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)
        
        # Get probability scores if available
        try:
            probability = model.predict_proba(features)[0][1]
        except:
            probability = None
        
        return {
            'prediction': int(prediction[0]),
            'probability': probability
        }
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return {
            'error': str(e),
            'prediction': None,
            'probability': None
        } 