Real-Time Drowsiness Detection using MobileNetV2

This project implements a real-time drowsiness detection system using a deep learning approach based on the MobileNetV2 convolutional neural network architecture. The system monitors a person’s eye state through a webcam and classifies whether the eyes are open or closed. If prolonged eye closure is detected, an alert is triggered.

The model is trained using transfer learning with a labeled dataset consisting of two classes: awake (eyes open) and sleepy (eyes closed). MobileNetV2 is used as a feature extractor, and a custom classification layer is added for binary classification.

During real-time execution, the system captures video frames using OpenCV, detects the face region, extracts the relevant eye area, and performs classification using the trained model. A time-based threshold is applied to avoid false alerts caused by normal blinking.

Key Features

Deep learning-based eye state classification

Transfer learning using MobileNetV2

Real-time webcam monitoring

Binary classification (Open / Closed eyes)

Alert mechanism for sustained eye closure

Lightweight architecture suitable for real-time inference

Technologies Used

Python

TensorFlow / Keras

MobileNetV2 (Convolutional Neural Network)

OpenCV

NumPy

Winsound or Pygame for alert generation

Model Performance

Training Accuracy: Approximately 96%

Test Accuracy: Approximately 91%

Optimized for real-time prediction

Applications

General fatigue monitoring

Attention monitoring systems

Workplace or study alert systems

Human state analysis in safety-critical environments

Future Improvements

Integration of LSTM for temporal sequence modeling

Improved robustness under varying lighting conditions

Enhanced motion stability

Deployment on embedded or edge devices# real-time-drowsiness-detection
Real-time deep learning-based drowsiness detection system using MobileNetV2 for eye state classification with webcam monitoring and alert mechanism.
