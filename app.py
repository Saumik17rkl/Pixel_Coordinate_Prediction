import os
import io
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

# -------------------------
# App initialization
# -------------------------

app = Flask(__name__)
CORS(app)  # allow all origins (tighten in prod)

# -------------------------
# Configuration
# -------------------------

MODEL_PATH = os.getenv("MODEL_PATH", "pixel_coordinate_regressor.keras")
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "50"))
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

model = tf.keras.models.load_model(MODEL_PATH)

# -------------------------
# Signal-preserving cleaning
# -------------------------

def clean_image_preserve_signal(image: np.ndarray) -> np.ndarray:
    """
    Preserve only the strongest pixel (signal invariant).
    """
    cleaned = np.zeros_like(image)
    y, x = np.unravel_index(np.argmax(image), image.shape)
    cleaned[y, x] = 255
    return cleaned

# -------------------------
# Preprocessing
# -------------------------

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Clean, normalize, and reshape image for inference.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image = np.array(image, dtype=np.uint8)

    # Clean noisy pixels
    image = clean_image_preserve_signal(image)

    # Normalize
    image = image.astype("float32") / 255.0

    # Add batch & channel dims → (1, 50, 50, 1)
    image = image[np.newaxis, ..., np.newaxis]

    return image

# -------------------------
# Decode heatmap
# -------------------------

def decode_heatmap(heatmap: np.ndarray):
    """
    Convert heatmap to (x, y) pixel coordinates.
    """
    heatmap = heatmap.squeeze()
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return int(x), int(y)

# -------------------------
# Health check
# -------------------------

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "API running"})

# -------------------------
# Inference endpoint
# -------------------------

@app.route("/predict", methods=["POST"])
def predict_pixel_coordinates():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if not file.mimetype.startswith("image/"):
        return jsonify({"error": "Invalid image file"}), 400

    image_bytes = file.read()
    input_tensor = preprocess_image(image_bytes)

    # Predict heatmap
    heatmap = model.predict(input_tensor, verbose=0)[0]

    # Decode coordinates
    x, y = decode_heatmap(heatmap)

    return jsonify({
        "predicted_x": x,
        "predicted_y": y
    })

# -------------------------
# Run server
# -------------------------

if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=False)
