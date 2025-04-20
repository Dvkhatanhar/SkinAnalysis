import cv2
import numpy as np
from tensorflow.keras.models import load_model

def load_trained_model():
    model = load_model("model/KlasifikasiWajah-pest-65.23.h5")
    return model

def preprocess_image(image_path):
    """
    Load and preprocess the image to match model input size (150x150).
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 150))  # 👈 Resize to expected input
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_model(image_array, model):
    """
    Predict the class probabilities.
    """
    y_probs = model.predict(image_array, verbose=0)
    return y_probs
