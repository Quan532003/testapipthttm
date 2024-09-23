from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the model and feature names
model = joblib.load('linear_regression_model.pkl')
feature_names = joblib.load('feature_names.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_weight():
    try:
        data = request.json
        height = float(data['height'])
        job = data['job'].strip()

        # One-hot encoding for job
        job_encoded = [0] * (len(feature_names) - 1)  # Adjust based on the number of job categories
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
            'salesperson': 12  # Add more jobs as necessary
        }

        # Check job and encode
        if job in job_dict:
            job_encoded[job_dict[job]] = 1
        else:
            return jsonify({'error': "Unknown job type. Please enter a valid job."}), 400

        # Create DataFrame for input
        input_data = pd.DataFrame([[height] + job_encoded], columns=feature_names)

        # Predict weight
        weight_pred = model.predict(input_data)[0]

        return jsonify({'predicted_weight': round(weight_pred, 2)})

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'An error occurred during prediction.'}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
