from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model pipeline
model_pipeline = joblib.load('car_price_predictor_model.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        company = request.form.get('company')
        year = request.form.get('year')
        kms_driven = request.form.get('kms_driven')
        fuel_type = request.form.get('fuel_type')

        # Validate inputs
        if not all([company, year, kms_driven, fuel_type]):
            return render_template('index.html', prediction="Please fill all fields.")

        # Create input DataFrame
        year = pd.to_numeric(year, errors='coerce')
        kms_driven = pd.to_numeric(kms_driven, errors='coerce')

        if pd.isna(year) or pd.isna(kms_driven):
            return render_template('index.html', prediction="Invalid year or kms driven.")

        input_data = pd.DataFrame({
            'company': [company],
            'year': [year],
            'kms_driven': [kms_driven],
            'fuel_type': [fuel_type]
        })

        # Feature engineering (if model expects car_age instead of year)
        input_data['car_age'] = 2025 - input_data['year']
        input_data = input_data.drop(columns=['year'])

        # Predict
        predicted_price = model_pipeline.predict(input_data)[0]

        return render_template('index.html', prediction=f"Estimated Price: ₹{round(predicted_price, 2)}")

    except Exception as e:
        # Optional: log the error using logging module
        return render_template('index.html', prediction=f"Error: {str(e)}")

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
