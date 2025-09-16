import requests

url = 'http://localhost:8000/predict'

customer_id = 'HBN - 123'

data = {
  "car_name": "Hyundai Grand",
  "brand": "Hyundai",
  "model": "Grand",
  "vehicle_age": 5,
  "km_driven": 20000,
  "seller_type": "Individual",
  "fuel_type": "Petrol",
  "transmission_type": "Manual",
  "mileage": 18.9,
  "engine": 1197,
  "max_power": 82,
  "seats": 5
}
response = requests.post(url, json=data).json()
print(response)