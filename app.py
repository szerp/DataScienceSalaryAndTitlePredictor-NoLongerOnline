from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)  # Ignore UserWarning messages

app = Flask(__name__)

# Load saved models and encoders
salary_model = joblib.load('salary_prediction_model.pkl')
title_model = joblib.load('title_prediction_model.pkl')
location_encoder = joblib.load('location_encoder.pkl')
mlb = joblib.load('mlb.pkl') if joblib.os.path.exists('mlb.pkl') else MultiLabelBinarizer()

# Function to preprocess input data
def preprocess_input(form_data):
    location = form_data['location']
    skills = [form_data.get(f'skill_{i}') for i in range(1, 6) if form_data.get(f'skill_{i}')]
    experience = int(form_data['years_of_experience'])  # Assuming 'years_of_experience' is the field name in your form
    return location, skills, experience

@app.route('/')
def index():
    specified_skills = [
        'python', 'machine learning', 'sql', 'aws', 'azure', 'gcp', 'docker', 'etl',
        'git', 'kafka', 'airflow', 'nosql', 'pandas', 'pyspark', 'tableau',
        'deep learning', 'tensorflow', 'pytorch', 'hadoop', 'numpy', 'apache spark',
        'mysql', 'postgresql', 'spark ml'
    ]
    return render_template('index.html', locations=location_encoder.classes_, skills=specified_skills, years_of_experience=range(1, 21))

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form
    location, skills, experience = preprocess_input(form_data)
    location_encoded = location_encoder.transform([location]).reshape(1, -1)
    skills_encoded = mlb.transform([skills])
    experience_array = np.array([[experience]])  # Ensure this is shaped as (1, 1)

    # Ensure correct concatenation along axis 1 (columns)
    X = np.hstack((location_encoded, skills_encoded, experience_array))

    predicted_salary = salary_model.predict(X)
    predicted_title = title_model.predict(X)

    return jsonify({'predicted_salary': predicted_salary[0], 'predicted_title': predicted_title[0]})


if __name__ == '__main__':
    app.run(debug=True)
