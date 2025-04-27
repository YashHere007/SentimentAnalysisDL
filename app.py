from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pickle
import re
import string
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize Flask app
app = Flask(__name__, static_folder='static')
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Health check endpoint
@app.route('/health', methods=['GET', 'HEAD'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

# Serve frontend
@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static_files(path):
    return send_from_directory(app.static_folder, path)

# Load tokenizer, label encoder, and TFLite model
try:
    with open('sentiment_model_bundle.pkl', 'rb') as f:
        bundle = pickle.load(f)
    tokenizer = bundle['tokenizer']
    label_encoder = bundle['label_encoder']
    max_len = bundle['max_len']

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

except Exception as e:
    print(f"[ERROR] Problem during model loading: {e}")
    raise

# Function to clean text
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

# Predict emotion using TFLite model
def predict_emotion(text):
    paragraph_clean = clean_text(text)
    sequence = tokenizer.texts_to_sequences([paragraph_clean])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post')
    
    # Add batch dimension
    padded_sequence = np.expand_dims(padded_sequence, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], padded_sequence)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]

    predicted_class_index = np.argmax(prediction)
    predicted_label = label_encoder.inverse_transform([predicted_class_index])[0]
    predicted_confidence = prediction[predicted_class_index]

    # Emotion Mapping
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

# API Endpoint
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

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
