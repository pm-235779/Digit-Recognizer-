

import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
import cv2
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
model = tf.keras.models.load_model("digit_model.h5")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    try:
        # Extract and decode the base64 image
        image_data = data['image'].split(',')[1]
        image = Image.open(BytesIO(base64.b64decode(image_data))).convert('L')
        image = image.resize((28, 28))
        image = np.array(image)
        image = 255 - image  # Invert for white digit on black background
        image = image / 255.0
        image = image.reshape(1, 28, 28, 1)

        # Predict
        predictions = model.predict(image)
        predicted_digit = int(np.argmax(predictions))
        confidence = float(np.max(predictions))

        return jsonify({
            'prediction': predicted_digit,
            'confidence': round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
