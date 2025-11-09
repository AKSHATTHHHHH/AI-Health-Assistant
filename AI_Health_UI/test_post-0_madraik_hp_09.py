import requests
host="localhost",
user="root",
password="Freefire@113",
database="ai_health_db"
port=3306 


# Paste the certificate content directly here
ca_cert = """
-----BEGIN CERTIFICATE-----
<copy everything from ca.pem here exactly>
-----END CERTIFICATE-----
"""


url = "https://postman-echo.com/post"

data = {
    "age": 50,
    "sex": 1,
    "cp": 0,
    "trestbps": 130,
    "chol": 250,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 1,
    "ca": 0,
    "thal": 2,
    "prediction": 1,
    "confidence": 0.87,
    "rule_diagnosis": "Likely Heart Disease"
}

response = requests.post(url, json=data)
print(response.status_code)
print(response.json())