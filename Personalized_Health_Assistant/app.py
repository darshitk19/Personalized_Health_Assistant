from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS 

app = Flask(__name__)
CORS(app) 

# Load the model, vectorizer, and label encoder
model = joblib.load('disease_prediction_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/', methods=['GET'])
def home():
    return "Request was received on this route"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        symptom_text = data.get('symptom', '')

        if not symptom_text:
            return jsonify({'error': 'No symptom provided'}), 400
        
        # Vectorize the symptom text
        symptom_vectorized = vectorizer.transform([symptom_text])
        
        # Predict disease
        prediction = model.predict(symptom_vectorized)
        predicted_disease = label_encoder.inverse_transform(prediction)
        
        return jsonify({'predicted_disease': predicted_disease[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
