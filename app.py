import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import librosa
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Path to the trained h5 model
MODEL_PATH = 'Deep Learning/SER_model.h5'

# Load the model directly during app initialization
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

def convertclasstoemotion(pred):
    label_conversion = {
        '0': 'neutral',
        '1': 'calm',
        '2': 'happy',
        '3': 'sad',
        '4': 'angry',
        '5': 'fearful',
        '6': 'disgust',
        '7': 'surprised'
    }
    return label_conversion.get(str(pred), "Unknown")

def predict_audio(filepath):
    if model is None:
        raise Exception("Model not loaded. Check server logs.")
        
    data, sampling_rate = librosa.load(filepath)
    mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
    
    # Expand dimensions twice as expected by the model input
    x = np.expand_dims(mfccs, axis=1)
    x = np.expand_dims(x, axis=0)
    
    # Use predict instead of predict_classes (as it's removed in newer TF versions)
    predictions = model.predict(x)
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    return convertclasstoemotion(predicted_class)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file part'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            emotion = predict_audio(filepath)
            # Clean up the uploaded file after prediction
            os.remove(filepath) 
            return jsonify({'emotion': emotion})
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
