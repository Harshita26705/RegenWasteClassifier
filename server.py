from flask import Flask, Response, jsonify
import os
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load your ML model
try:
    model = tf.keras.models.load_model("waste_classifier.h5")  # Update path if needed
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Error loading model:", e)

# Initialize the camera (use a mock or handle errors for cloud environments)
camera = cv2.VideoCapture(0)  # 0 = default camera, change if needed

def classify_frame(frame):
    """Process frame and classify waste."""
    frame = cv2.resize(frame, (224, 224))  # Resize to model input size
    frame = np.expand_dims(frame, axis=0) / 255.0  # Normalize if needed
    prediction = model.predict(frame)
    return np.argmax(prediction)  # Assuming classification returns index

def generate_frames():
    """Continuously capture frames and classify."""
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            label = classify_frame(frame)  # Get ML prediction
            _, buffer = cv2.imencode(".jpg", frame)
            frame_bytes = buffer.tobytes()
            
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

@app.route("/")
def home():
    """Main route."""
    return "RegenWaste is running! <a href='/video_feed'>View Camera</a>"

@app.route("/video_feed")
def video_feed():
    """Stream live camera feed."""
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/classify", methods=["GET"])
def classify():
    """Classify waste from the latest camera frame."""
    success, frame = camera.read()
    if success:
        label = classify_frame(frame)
        return jsonify({"classification": str(label)})
    return jsonify({"error": "Camera not available"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Get port from environment variable
    app.run(host="0.0.0.0", port=port, debug=True)
