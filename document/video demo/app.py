from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("vgg16_model.h5")

class_labels = ['Biodegradable', 'Recyclable', 'Trash']  # Replace with your labels

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded!'

    f = request.files['file']
    upload_path = os.path.join('static/uploads', f.filename)
    f.save(upload_path)

    img = image.load_img(upload_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]

    return render_template('result.html', prediction=predicted_class, image_path=upload_path)

if __name__ == '__main__':
    app.run(debug=True)
