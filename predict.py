import pickle
import numpy as np
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model


#Load the trained model
loaded_model = load_model('nn_model.keras')
# Load the trained scaler
scaler = joblib.load('scaler.joblib')
#load the pipeline model
preprocessor = joblib.load(' preprocessor.joblib')

#do other stuff
app = Flask('Car')
@app.route("/")
def home():
    return "Car Prediction App is running"
@app.route("/predict", methods = ["POST"])
def predict():
    try:
        #get the JSON data from the api request
         data = request.get_json()
         input_data = pd.DataFrame([data])
         #check if input is provided
         if not data:
             return jsonify({"error": "Input not provided"}), 400

          #validate input columns
         required_columns = ('car_name', 'brand', 'model', 'vehicle_age', 'km_driven', 'seller_type',
                             'fuel_type', 'transmission_type', 'mileage', 'engine', 'max_power',
                             'seats')
         if not all(col in input_data.columns for col in required_columns):
            return jsonify({"error": f"Required columns missing. Required columns: {required_columns}"}), 400

         # 
         X = input_data[['brand', 'model', 'vehicle_age', 'km_driven', 'seller_type',
                             'fuel_type', 'transmission_type', 'mileage', 'engine', 'max_power',
                             'seats']]
         
         X = preprocessor.transform(X)
         num_cat_pre = X[:,:-1]
         model_pre = X[:,-1]
         #make prediction
         prediction = loaded_model.predict([num_cat_pre, model_pre])
         # Convert predictions back to original scale
         prediction = scaler.inverse_transform(prediction)

         #response
         result = {
        "Selling Price": float(prediction[0])}
         return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)