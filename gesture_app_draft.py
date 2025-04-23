#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pickle

# Load model and label encoder
model = tf.keras.models.load_model("pose_classifier.h5")
with open('pose_detection.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

st.title("Gesture/Emotion Classification")
st.text("Allow webcam to start prediction...")

# Setup MediaPipe
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False)

# Start webcam
frame_window = st.image([])
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to grab frame")
        break

    # Convert to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process with MediaPipe
    result = holistic.process(img_rgb)

    # Collect landmarks
    landmarks = []

    def extract_landmarks(landmark_list):
        if landmark_list:
            for lm in landmark_list.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0, 0.0, 0.0] * 33)  # fill dummy if missing

    extract_landmarks(result.pose_landmarks)
    extract_landmarks(result.left_hand_landmarks)
    extract_landmarks(result.right_hand_landmarks)
    extract_landmarks(result.face_landmarks)

    if len(landmarks) == model.input_shape[1]:
        # Predict
        input_data = np.array(landmarks).reshape(1, -1)
        prediction = model.predict(input_data)
        pred_label = label_encoder.inverse_transform([np.argmax(prediction)])
        st.write(f"Prediction: {pred_label[0]}")
    else:
        st.write("Incomplete landmarks detected, skipping prediction...")

    # Display
    frame_window.image(img_rgb)

cap.release()

