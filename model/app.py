#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pickle
import numpy as np
from PIL import Image

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import tensorflow as tf
tf.get_logger().setLevel("ERROR")


# In[2]:


# Dossier où se trouvent les fichiers générés
WORKING_DIR = r"Imagetest\testpfaaveckaggle8kprojetcopie"



MODEL_PATH = os.path.join(WORKING_DIR, "model.keras")   # ou model.h5
TOKENIZER_PATH = os.path.join(WORKING_DIR, "tokenizer.pkl")

max_length = 34  # ⚠️ le même que lors de l'entraînement


# In[3]:


model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

print("✅ Modèle et tokenizer chargés")


# In[4]:


vgg = VGG16(weights="imagenet")

feature_extractor = Model(
    inputs=vgg.inputs,
    outputs=vgg.get_layer("fc2").output
)


# In[5]:


from io import BytesIO

from tensorflow.keras.preprocessing.image import load_img, img_to_array

from io import BytesIO
from tensorflow.keras.preprocessing.image import img_to_array

def extract_features(image_bytes):
    """
    Convertit des bytes d'image en features pour VGG16
    """
    # Ouvrir l'image depuis les bytes et convertir en RGB
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    
    # Redimensionner pour VGG16
    image = image.resize((224, 224))
    
    # Convertir en array et ajouter la dimension batch
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    
    # Prétraitement VGG16
    image = preprocess_input(image)

    # Extraction des features
    feature = feature_extractor.predict(image, verbose=0)
    return feature



def convert_to_word(number, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == number:
            return word
    return None


def predict_caption(model, image_feature, tokenizer, max_length):
    print("🔹 predict_caption appelé")
    in_text = "startseq"

    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')

        yhat = model.predict([image_feature, sequence], verbose=0)
        yhat = np.argmax(yhat)

        word = convert_to_word(yhat, tokenizer)

        if word is None or word == "endseq":
            break

        in_text += " " + word
        print(f"🔄 mot ajouté: {word}")

    print(f"✅ Texte final pré-caption: {in_text}")
    return in_text



# In[6]:


def generate_caption_from_image(image_bytes):
    features = extract_features(image_bytes)
    caption = predict_caption(model, features, tokenizer, max_length)

    caption = caption.replace("startseq", "").replace("endseq", "").strip()
    return caption


# In[6]:


from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    print("📥 /predict appelée")

    if 'image' not in request.files:
        print("❌ image absente")
        return jsonify({'error': 'No image'}), 400

    image = request.files['image']
    print("✅ image reçue :", image.filename)

    # ⚠️ Correction : lire les bytes et utiliser la fonction correcte
    image_bytes = image.read()
    caption = generate_caption_from_image(image_bytes)

    print("📝 caption :", caption)

    return jsonify({'caption': caption})



# In[ ]:



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)



