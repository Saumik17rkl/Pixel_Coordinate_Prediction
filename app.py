from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import tensorflow as tf
from PIL import Image
import io

# -------------------------
# App initialization
# -------------------------

app = FastAPI(title="Pixel Coordinate Heatmap API")

MODEL_PATH = "pixel_coordinate_regressor.keras"
IMAGE_SIZE = 50

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
