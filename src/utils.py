import cv2
import winsound
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import pygame

# Load trained model
def load_trained_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model


# Preprocess frame for MobileNetV2
def preprocess_frame(frame):
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


# Predict awake or sleepy
def predict_state(model, frame):
    processed = preprocess_frame(frame)
    prediction = model(processed, training=False)[0][0].numpy()
    print("Prediction value:", prediction)


    if prediction > 0.5:
        return "sleepy", prediction
    else:
        return "awake", prediction


# Initialize alarm
def init_alarm():
    pygame.mixer.init()


# Play alarm
def play_alarm(sound_path=None):
    winsound.Beep(1000, 1000)


# Stop alarm
def stop_alarm():
    pygame.mixer.music.stop()
