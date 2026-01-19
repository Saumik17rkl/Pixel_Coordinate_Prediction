import os
from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import tensorflow as tf
from PIL import Image
import io
from fastapi.middleware.cors import CORSMiddleware

# -------------------------
# App initialization
# -------------------------

app = FastAPI(
    title="Pixel Coordinate Heatmap API",
    description="API for predicting pixel coordinates from images",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
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
    Preserve the strongest pixel only.
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
    Clean, normalize, and reshape image.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    image = np.array(image, dtype=np.uint8)

    # Clean noisy pixels
    image = clean_image_preserve_signal(image)

    # Normalize
    image = image.astype("float32") / 255.0

    # Add batch + channel dims
    image = image[np.newaxis, ..., np.newaxis]

    return image


# -------------------------
# Decode heatmap
# -------------------------

def decode_heatmap(heatmap: np.ndarray):
    """
    Convert heatmap to (x, y).
    """
    heatmap = heatmap.squeeze()  # (50, 50)
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return int(x), int(y)


# -------------------------
# Health check
# -------------------------

@app.get("/")
def root():
    return {"status": "API running"}


# -------------------------
# Inference endpoint
# -------------------------

@app.post("/predict")
async def predict_pixel_coordinates(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file")

    image_bytes = await file.read()
    input_tensor = preprocess_image(image_bytes)

    # Predict heatmap
    heatmap = model.predict(input_tensor, verbose=0)[0]

    # Decode coordinates
    x, y = decode_heatmap(heatmap)

    return {
        "predicted_x": x,
        "predicted_y": y
    }
