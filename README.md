# MEDISYNC AI - Medical Diagnosis Platform

MEDISYNC AI is a comprehensive medical diagnosis platform that leverages machine learning algorithms to provide preliminary diagnoses for various medical conditions. The platform currently supports breast cancer and lung cancer prediction using multiple machine learning models.

## Features

- **Multiple ML Models**: Support for various machine learning algorithms including:
  - SVM (RBF and Linear kernels)
  - K-Nearest Neighbors (KNN)
  - Decision Trees
  - Random Forest
  - Logistic Regression
  - Gradient Boosting

- **User Authentication**: Secure login and registration system
- **Interactive UI**: Modern and responsive web interface
- **Detailed Predictions**: Comprehensive analysis with confidence scores
- **Health Recommendations**: Personalized health advice based on predictions

## Project Structure

```
MEDISYNC-AI/
├── MEDISYNC-AI-main/          # Frontend files
│   ├── assets/                # Images and static assets
│   ├── *.html                 # HTML pages
│   └── style.css              # Main stylesheet
│
├── Model/                     # Backend ML models
│   ├── breast_cancer/         # Breast cancer prediction models
│   │   ├── model.py          # Model interface
│   │   ├── *.joblib          # Trained model files
│   │   └── *.csv             # Dataset
│   │
│   ├── lung_cancer/          # Lung cancer prediction models
│   │   ├── model.py          # Model interface
│   │   ├── *.pkl             # Trained model files
│   │   └── *.csv             # Dataset
│   │
│   └── models.py             # Main model interface
│
├── app.py                    # Flask application
├── requirements.txt          # Python dependencies
└── users.db                 # User database
```

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

## Installation

1. Clone the repository:
```bash
git clone https://github.com/fred-rick20/MEDISYNC-AI.git
cd MEDISYNC-AI
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

## Model Training

The models are pre-trained and saved in their respective directories. If you want to retrain the models:

1. For Breast Cancer:
```bash
cd Model/breast_cancer
python Breast_cancer_prediction_final.py
```

2. For Lung Cancer:
```bash
cd Model/lung_cancer
python lung-cancer-prediction-98.py
```

## Usage

1. Register a new account or login
2. Navigate to the AI Diagnosis section
3. Select the type of diagnosis (Breast Cancer or Lung Cancer)
4. Enter the required measurements/parameters
5. Select the preferred ML algorithm
6. Submit for prediction
7. View the results and recommendations

## Model Accuracies

### Breast Cancer Prediction
- SVM (RBF Kernel): 96.49%
- SVM (Linear Kernel): 99.12%
- K-Nearest Neighbors: 94.74%
- Decision Tree: 88.60%
- Random Forest: 98.25%
- Logistic Regression: 96.49%

### Lung Cancer Prediction
- SVM: 98.00%
- KNN: 97.00%
- Gradient Boosting: 97.00%

## Contributing

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset sources:
  - Breast Cancer: Wisconsin Breast Cancer Dataset
  - Lung Cancer: Survey Lung Cancer Dataset
- Machine Learning libraries: scikit-learn, pandas, numpy
- Web Framework: Flask
- Frontend: HTML, CSS, JavaScript
