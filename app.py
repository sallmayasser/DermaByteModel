from flask import Flask, request, jsonify
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import urllib.request

app = Flask(__name__)


def load_image_from_url(url):
    # Download the image from the URL
    img_data = urllib.request.urlopen(url).read()
    # Convert the bytes to a numpy array
    nparr = np.frombuffer(img_data, np.uint8)
    # Decode the numpy array to an image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def predict_skin_disease(image_url):
    class_names = ['nail-fungus', 'ringworm', 'shingles', 'impetigo',
                   'athlete-foot', 'chickenpox', 'cutaneous-larva-migrans', 'cellulitis']
    m = tf.keras.models.load_model(
        '/content/drive/MyDrive/Colab Notebooks/final model/derma.h5', custom_objects={'KerasLayer': hub.KerasLayer})
    img = load_image_from_url(image_url)
    img_resized = cv2.resize(img, (224, 224))
    ins = np.array([img_resized]) / 255.0
    pred = m.predict(ins)
    predicted_class_index = np.argmax(pred)
    predicted_class_name = class_names[predicted_class_index]
    return predicted_class_name


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json()
    image_url = data['image_url']
    predicted_class = predict_skin_disease(image_url)
    return jsonify({'predicted_class': predicted_class})


if __name__ == '__main__':
    app.run(debug=True)
    print('Server Running')
