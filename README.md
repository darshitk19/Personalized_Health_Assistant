# Personalized Health Assistant

## 📌 Overview
The **Personalized Health Assistant** is a machine learning-based application that predicts diseases based on user symptoms. It includes a Flask-based API and a frontend UI for easy user interaction. The model leverages **TF-IDF vectorization** and a **trained classification model** for accurate disease prediction.

## 🚀 Features
- Accepts user symptoms as text input.
- Processes input using a **TF-IDF vectorizer**.
- Uses a trained **machine learning model** for disease prediction.
- Provides predictions via a **Flask API**.
- Includes a **web-based UI** for user interaction.

## 📂 Project Structure
```
📁 Personalized_Health_Assistant/
│──
│── 
│── app.py                     # Flask API for disease prediction
│── index.html                  # Frontend UI for user interaction
│── disease_prediction_model.pkl # Trained machine learning model
│── tfidf_vectorizer.pkl         # TF-IDF vectorizer for text processing
│── label_encoder.pkl            # Label encoder for mapping predictions
│── README.md                    # Project documentation
```

## ⚡ Installation & Usage

### 1️⃣ Install Dependencies
```bash
pip install flask joblib scikit-learn numpy
```

### 2️⃣ Run the Flask Server
```bash
python app.py
```
The server will start at: **http://127.0.0.1:5000**

### 3️⃣ Open the Web Interface
1. Open `index.html` in a web browser.
2. Enter symptoms and click the **Predict** button.

### 4️⃣ Test the API
Use `curl` or Python to send a request:
```python
import requests

url = "http://127.0.0.1:5000/predict"
data = {"symptom": "I have a runny nose and I am sneezing all the time."}

response = requests.post(url, json=data)
print(response.json())  # Expected Output: {'predicted_disease': 'Common Cold'}
```

## 🔥 API Endpoints
| Endpoint  | Method | Description |
|-----------|--------|-------------|
| `/`       | GET    | Health Assistant Home Page |
| `/predict` | POST  | Predict disease from symptoms |

## 👨‍💻 Author
- **Darshit** - Developer & ML Engineer


