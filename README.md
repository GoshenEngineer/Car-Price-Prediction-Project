# Car-Price-Prediction-Project
This project is a machine learning web API that predicts the selling price of used cars based on various features like brand, model, mileage, engine, and more.
It is built using TensorFlow/Keras, Flask, and deployed with Docker.

# Project Structure
.
├── cardekho_dataset.csv
├── Car_Price_Prediction.ipynb
├── Car_Price_Prediction.py
├── predict.py
├── nn_model.keras
├── preprocessor.joblib
├── scaler.joblib
├── Pipfile
├── Pipfile.lock
└── Dockerfile
# Features
Preprocesses categorical and numerical data using ColumnTransformer
Deep learning regression model built with TensorFlow/Keras
Scales predictions back to original price values using StandardScaler
Flask API endpoint /predict to serve predictions
Dockerized for easy deployment

# Setup & Run

1. Install dependencies

pip install pipenv
pipenv install --deploy --system


2. Run locally

python predict.py
# App runs at http://127.0.0.1:8000

3. Or run with Docker

docker build -t car-prediction .
docker run -p 8000:8000 car-prediction

# API Usage
POST /predict
Request JSON:
{
  "car_name": "Honda City",
  "brand": "Honda",
  "model": "City",
  "vehicle_age": 5,
  "km_driven": 45000,
  "seller_type": "Dealer",
  "fuel_type": "Petrol",
  "transmission_type": "Manual",
  "mileage": 18.0,
  "engine": 1497,
  "max_power": 117.0,
  "seats": 5
}
Response:
{ "Selling Price": 550000.0 }

# Notes
preprocessor.joblib transforms input features
nn_model.keras predicts price
scaler.joblib inverses scaling on predictions


# License
This project is for educational purposes.
