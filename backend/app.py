from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import re
import string
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # Adjust origins for production

# Health check endpoint for Render
@app.route('/', methods=['GET', 'HEAD'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

# Load the model bundle
try:
    with open('sentiment_model_bundle.pkl', 'rb') as f:
        model_bundle = pickle.load(f)
    model = model_bundle['model']
    tokenizer = model_bundle['tokenizer']
    label_encoder = model_bundle['label_encoder']
    max_len = model_bundle['max_len']
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Preprocessing function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# Prediction function
def predict_emotion(text):
    paragraph_clean = clean_text(text)
    sequence = tokenizer.texts_to_sequences([paragraph_clean])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post')
    prediction = model.predict(padded_sequence, verbose=0)[0]
    predicted_class_index = np.argmax(prediction)
    predicted_label = label_encoder.inverse_transform([predicted_class_index])[0]
    predicted_confidence = prediction[predicted_class_index]

    if predicted_label == 'Positive':
        if predicted_confidence > 0.95:
            return "Extreme Happy"
        else:
            return "Happy"
    elif predicted_label == 'Neutral':
        return "Normal"
    elif predicted_label == 'Negative':
        if predicted_confidence < 0.5:
            return "Extreme Sad"
        else:
            return "Sad"
    else:
        return "Normal"

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        emotion = predict_emotion(text)
        return jsonify({'emotion': emotion})
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)