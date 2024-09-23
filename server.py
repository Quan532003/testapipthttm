from flask import Flask, render_template, request, redirect, url_for, flash
import joblib
import pandas as pd

# Load the model and feature names
model = joblib.load('D:/Ky1Nam4/PTCHTTM/BT5/linear_regression_model.pkl')
feature_names = joblib.load('D:/Ky1Nam4/PTCHTTM/BT5/feature_names.pkl')

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a random secret key for production

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        height_str = request.form['height'].replace(',', '.')
        job = request.form['job'].strip()
        try:
            height = float(height_str)

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
                raise ValueError("Unknown job type. Please enter a valid job.")

            # Create DataFrame for input
            input_data = pd.DataFrame([[height] + job_encoded], columns=feature_names)

            # Predict weight
            weight_pred = model.predict(input_data)[0]
            result = f"Predicted Weight: {weight_pred:.2f} kg"

        except ValueError as e:
            flash(str(e))

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
