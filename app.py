import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

st.set_page_config(layout="wide")
st.title("🎓 Student Monitoring - PoseNet Cheating Detection")

# Load PoseNet from TF Hub
@st.cache_resource
def load_model():
    model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
    return model.signatures['serving_default']

model = load_model()

# Detect pose keypoints
def detect_pose(frame):
    input_image = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), 192, 192)
    input_image = tf.cast(input_image, dtype=tf.int32)
    outputs = model(input_image)
    keypoints = outputs['output_0'].numpy()
    return keypoints

# Cheating logic
def detect_cheating(keypoints, threshold=0.3):
    kp = keypoints[0, 0, :, :]
    nose, left_eye, right_eye, left_ear, right_ear = kp[0], kp[1], kp[2], kp[3], kp[4]

    alerts = []

    if nose[2] < threshold:
        alerts.append("❌ Face not visible")
    if left_eye[2] < threshold and right_eye[2] < threshold:
        alerts.append("❌ Eyes not visible")
    if left_ear[2] < threshold and right_ear[2] < threshold:
        alerts.append("❌ Possibly looking away")

    return alerts

# Start video
stframe = st.empty()
run = st.checkbox("Start Monitoring")

if run:
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture video.")
            break

        # Run pose detection
        keypoints = detect_pose(frame)

        # Run cheating detection
        alerts = detect_cheating(keypoints)

        # Show alerts
        for idx, alert in enumerate(alerts):
            cv2.putText(frame, alert, (10, 30 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)

        # Display frame
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        if not run:
            break

    cap.release()
    cv2.destroyAllWindows()
