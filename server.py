from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the model and feature names
model = joblib.load('linear_regression_model.pkl')
feature_names = joblib.load('feature_names.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    height = data.get('height')
    job = data.get('job')
    
    if height is None or job is None:
        return jsonify({"error": "Missing data"}), 400

    try:
        height = float(height)

        # One-hot encoding for job
        job_encoded = [0] * (len(feature_names) - 1)
        job_dict = {
            'musician': 0,
            'writer': 1,
            'nurse': 2,
            'researcher': 3,
            'manager': 4,
            'photographer': 5,
            'veterinarian': 6,
            'technician': 7,
            'artist': 8,
            'carpenter': 9,
            'teacher': 10,
            'dentist': 11,
            'salesperson': 12
        }

        if job in job_dict:
            job_encoded[job_dict[job]] = 1
        else:
            return jsonify({"error": "Unknown job type"}), 400

        # Prepare input data
        input_data = pd.DataFrame([[height] + job_encoded], columns=feature_names)
        weight_pred = model.predict(input_data)[0]

        return jsonify({"predicted_weight": weight_pred})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
