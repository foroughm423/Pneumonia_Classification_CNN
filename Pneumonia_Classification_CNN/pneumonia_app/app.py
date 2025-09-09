from flask import Flask, request, render_template
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('best_model.keras')

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    prediction = None
    if 'file' not in request.files:
        return render_template('index.html', prediction=" No file uploaded!")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction=" No file selected!")
    
    try:
        # Open and preprocess the uploaded image
        img = Image.open(io.BytesIO(file.read())).convert('L')  # Grayscale
        img = img.resize((150, 150))  
        img_array = np.array(img) / 255.0  
        img_array = np.expand_dims(img_array, axis=0)  
        img_array = np.expand_dims(img_array, axis=-1)  

        # Prediction
        pred = model.predict(img_array)
        result = ' PNEUMONIA' if pred[0][0] > 0.5 else ' NORMAL'
        prediction = f"{result}"
    except Exception as e:
        prediction = f" Error processing image: {str(e)}"
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)