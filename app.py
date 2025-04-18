from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for, session, flash
import numpy as np
import os
from Model.models import predict
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import sqlite3
import secrets
from Model.lung_cancer.model import predict as predict_lung_cancer
from Model.breast_cancer.model import predict as predict_breast_cancer

app = Flask(__name__, static_folder='MEDISYNC-AI-main', template_folder='MEDISYNC-AI-main')
app.secret_key = secrets.token_hex(16)  # Generate a secure secret key

# Context processor to make user info available to all templates
@app.context_processor
def inject_user():
    return {
        'is_authenticated': 'user_id' in session,
        'user_name': session.get('user_name', None)
    }

# Dictionary of algorithms and their accuracies
ALGORITHMS = {
    'svm': {'name': 'Support Vector Machine', 'accuracy': '98%'},
    'knn': {'name': 'K-Nearest Neighbors', 'accuracy': '78%'},
    'logistic': {'name': 'Logistic Regression', 'accuracy': '85%'},
    'random_forest': {'name': 'Random Forest', 'accuracy': '88%'},
    'gradient_boosting': {'name': 'Gradient Boosting', 'accuracy': '88%'}
}

# Available algorithms for each model
LUNG_CANCER_ALGORITHMS = {
    'svm': {'name': 'Support Vector Machine', 'accuracy': '98%'},
    'knn': {'name': 'K-Nearest Neighbors', 'accuracy': '96%'},
    'gradient_boosting': {'name': 'Gradient Boosting', 'accuracy': '97%'}
}

BREAST_CANCER_ALGORITHMS = {
    'svm_rbf': {'name': 'SVM (RBF Kernel)', 'accuracy': '96.49%'},
    'svm_linear': {'name': 'SVM (Linear Kernel)', 'accuracy': '99.12%'},
    'knn': {'name': 'K-Nearest Neighbors', 'accuracy': '94.74%'},
    'decision_tree': {'name': 'Decision Tree', 'accuracy': '88.60%'},
    'random_forest': {'name': 'Random Forest', 'accuracy': '98.25%'},
    'logistic': {'name': 'Logistic Regression', 'accuracy': '96.49%'}
}

# Database initialization
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return send_from_directory('MEDISYNC-AI-main', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('MEDISYNC-AI-main', path)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if not all([first_name, last_name, email, password, confirm_password]):
            return render_template('signup.html', error="All fields are required")

        if password != confirm_password:
            return render_template('signup.html', error="Passwords do not match")

        hashed_password = generate_password_hash(password)

        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute('INSERT INTO users (first_name, last_name, email, password) VALUES (?, ?, ?, ?)',
                     (first_name, last_name, email, hashed_password))
            conn.commit()
            conn.close()

            return render_template('login.html', success="Account created successfully! Please log in.")
        except sqlite3.IntegrityError:
            return render_template('signup.html', error="Email already exists")
        except Exception as e:
            return render_template('signup.html', error=f"An error occurred: {str(e)}")

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        if not email or not password:
            return render_template('login.html', error="Email and password are required")

        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute('SELECT * FROM users WHERE email = ?', (email,))
            user = c.fetchone()
            conn.close()

            if user and check_password_hash(user[4], password):
                session['user_id'] = user[0]
                session['user_name'] = f"{user[1]} {user[2]}"
                next_page = request.args.get('next')
                return redirect(next_page or url_for('home'))
            else:
                return render_template('login.html', error="Invalid email or password")

        except Exception as e:
            return render_template('login.html', error=f"An error occurred: {str(e)}")

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route('/lung-cancer-prediction')
@login_required
def lung_cancer_prediction():
    return render_template('lung-cancer-prediction.html', algorithms=LUNG_CANCER_ALGORITHMS)

@app.route('/breast-cancer-prediction')
@login_required
def breast_cancer_prediction():
    return render_template('breast-cancer-prediction.html', algorithms=BREAST_CANCER_ALGORITHMS)

@app.route('/predict_lung_cancer', methods=['POST'])
@login_required
def predict_lung():
    try:
        # Get form data and selected algorithm
        data = request.form
        algorithm = data.get('algorithm', 'svm')
        
        # Convert form data to feature array
        features = [
            int(data['gender']), int(data['age']), int(data['smoking']),
            int(data['yellow_fingers']), int(data['anxiety']),
            int(data['peer_pressure']), int(data['chronic_disease']),
            int(data['fatigue']), int(data['allergy']),
            int(data['wheezing']), int(data['alcohol']),
            int(data['coughing']), int(data['shortness_of_breath']),
            int(data['swallowing_difficulty']), int(data['chest_pain'])
        ]
        
        # Make prediction
        result = predict_lung_cancer(features, algorithm)
        
        # Generate health message based on prediction
        if result['prediction'] == 1:
            health_message = """
                <p>Based on the provided symptoms, there may be an elevated risk of lung cancer. 
                Please consult with a healthcare professional for a thorough evaluation.</p>
                <ul>
                    <li>Schedule an appointment with your doctor</li>
                    <li>Discuss your symptoms and risk factors</li>
                    <li>Consider getting a chest X-ray or CT scan</li>
                </ul>
            """
        else:
            health_message = """
                <p>Based on the provided symptoms, the risk of lung cancer appears to be lower. 
                However, it's important to:</p>
                <ul>
                    <li>Maintain regular check-ups with your healthcare provider</li>
                    <li>Avoid smoking and exposure to secondhand smoke</li>
                    <li>Report any new or worsening symptoms to your doctor</li>
                </ul>
            """
        
        return render_template('lung-cancer-prediction.html',
                             result={'prediction': 'High Risk' if result['prediction'] == 1 else 'Low Risk',
                                     'probability': f"{result['probability']*100:.1f}%" if result['probability'] else None,
                                     'health_message': health_message,
                                     'algorithm': LUNG_CANCER_ALGORITHMS[algorithm]},
                             algorithms=LUNG_CANCER_ALGORITHMS)
                             
    except Exception as e:
        return render_template('lung-cancer-prediction.html',
                             error=f"An error occurred: {str(e)}",
                             algorithms=LUNG_CANCER_ALGORITHMS)

@app.route('/predict_breast_cancer', methods=['POST'])
@login_required
def predict_breast():
    try:
        # Get form data and selected algorithm
        data = request.form
        algorithm = data.get('algorithm', 'svm_rbf')
        
        # Convert form data to feature array
        features = [
            # Mean values
            float(data['radius_mean']), float(data['texture_mean']), 
            float(data['perimeter_mean']), float(data['area_mean']),
            float(data['smoothness_mean']), float(data['compactness_mean']),
            float(data['concavity_mean']), float(data['concave_points_mean']),
            float(data['symmetry_mean']), float(data['fractal_dimension_mean']),
            
            # Standard error values
            float(data['radius_se']), float(data['texture_se']),
            float(data['perimeter_se']), float(data['area_se']),
            float(data['smoothness_se']), float(data['compactness_se']),
            float(data['concavity_se']), float(data['concave_points_se']),
            float(data['symmetry_se']), float(data['fractal_dimension_se']),
            
            # Worst values
            float(data['radius_worst']), float(data['texture_worst']),
            float(data['perimeter_worst']), float(data['area_worst']),
            float(data['smoothness_worst']), float(data['compactness_worst']),
            float(data['concavity_worst']), float(data['concave_points_worst']),
            float(data['symmetry_worst']), float(data['fractal_dimension_worst'])
        ]
        
        # Make prediction
        result = predict_breast_cancer(features, algorithm)
        
        # Generate health message based on prediction
        if result['prediction'] == 1:
            health_message = """
                <p>Based on the diagnostic measurements, there may be indicators suggesting malignancy. 
                It is crucial to:</p>
                <ul>
                    <li>Consult with a specialist immediately</li>
                    <li>Schedule additional diagnostic tests</li>
                    <li>Discuss treatment options with your healthcare team</li>
                </ul>
            """
        else:
            health_message = """
                <p>Based on the diagnostic measurements, the indicators suggest benign characteristics. 
                However, it's important to:</p>
                <ul>
                    <li>Continue regular breast cancer screenings</li>
                    <li>Maintain routine check-ups with your healthcare provider</li>
                    <li>Be aware of any changes in breast tissue</li>
                </ul>
            """
        
        return render_template('breast-cancer-prediction.html',
                             result={'prediction': 'Malignant' if result['prediction'] == 1 else 'Benign',
                                     'probability': f"{result['probability']*100:.1f}%" if result['probability'] else None,
                                     'health_message': health_message,
                                     'algorithm': BREAST_CANCER_ALGORITHMS[algorithm]},
                             algorithms=BREAST_CANCER_ALGORITHMS)
                             
    except Exception as e:
        return render_template('breast-cancer-prediction.html',
                             error=f"An error occurred: {str(e)}",
                             algorithms=BREAST_CANCER_ALGORITHMS)

@app.route('/ai-diagnosis')
@login_required
def ai_diagnosis():
    return render_template('ai-diagnosis.html')

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html', user_name=session.get('user_name'))

if __name__ == '__main__':
    app.run(debug=True)
