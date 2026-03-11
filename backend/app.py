"""
FastAPI backend for handwritten digit recognition using a KNN model.
Serves predictions for base64-encoded canvas images.
"""

import os
import io
import base64
import pickle
import numpy as np
from scipy.ndimage import affine_transform
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Digit Recognition API",
    description="KNN-based handwritten digit recognition trained on MNIST",
    version="1.0.0",
)

# CORS — allow all origins so the GitHub Pages frontend can call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load model artifacts on startup
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

knn_model = None
pca_model = None
model_metadata = None


@app.on_event("startup")
def load_model():
    global knn_model, pca_model, model_metadata

    try:
        with open(os.path.join(MODEL_DIR, "knn_model.pkl"), "rb") as f:
            knn_model = pickle.load(f)

        with open(os.path.join(MODEL_DIR, "pca.pkl"), "rb") as f:
            pca_model = pickle.load(f)

        metadata_path = os.path.join(MODEL_DIR, "metadata.pkl")
        if os.path.exists(metadata_path):
            with open(metadata_path, "rb") as f:
                model_metadata = pickle.load(f)

        print("Model artifacts loaded successfully.")
    except FileNotFoundError as exc:
        print(f"WARNING: Could not load model artifacts: {exc}")
        print("Run train_model.py first to generate the model files.")


# ---------------------------------------------------------------------------
# Deskewing — must match training preprocessing exactly
# ---------------------------------------------------------------------------
def deskew(image):
    """
    Deskew a single 28x28 image using moment-based affine transformation.
    Identical to the function used during training.
    """
    img = image.reshape(28, 28)
    m = np.float64(img)
    total = m.sum()
    if total < 1e-10:
        return image

    grid_x, grid_y = np.mgrid[0:28, 0:28]
    cx = (grid_x * m).sum() / total
    cy = (grid_y * m).sum() / total

    mu20 = (grid_x * grid_x * m).sum() / total - cx * cx
    mu02 = (grid_y * grid_y * m).sum() / total - cy * cy
    mu11 = (grid_x * grid_y * m).sum() / total - cx * cy

    if mu02 < 1e-10:
        return image
    skew = mu11 / mu02

    transform = np.array([
        [1, 0],
        [skew, 1]
    ])

    offset = np.array([cx, cy]) - np.dot(transform, [14.0, 14.0])

    deskewed = affine_transform(
        img,
        transform,
        offset=offset,
        output_shape=(28, 28),
        order=1,
        mode='constant',
        cval=0.0
    )

    return deskewed.flatten()


# ---------------------------------------------------------------------------
# Image preprocessing pipeline
# ---------------------------------------------------------------------------
def preprocess_canvas_image(image_bytes: bytes) -> np.ndarray:
    """
    Convert a raw PNG image (from an HTML canvas) into a 784-element feature
    vector that matches the training preprocessing pipeline.

    Steps:
      1. Open image, convert to grayscale
      2. Resize to 28x28
      3. Invert if needed (canvas = white bg + black ink -> MNIST = black bg + white ink)
      4. Normalize to [0, 1]
      5. Center the digit by cropping to bounding box and re-centering in 20x20 area
      6. Deskew
    """
    # Open and convert to grayscale
    img = Image.open(io.BytesIO(image_bytes)).convert("L")

    # Resize to 28x28 using high-quality resampling
    img = img.resize((28, 28), Image.LANCZOS)

    # Convert to numpy array
    pixels = np.array(img, dtype=np.float64)

    # Invert colors: canvas has white background (255) with black drawing (0)
    # MNIST has black background (0) with white digits (255)
    pixels = 255.0 - pixels

    # Normalize to [0, 1] to match training
    pixels = pixels / 255.0

    # Flatten to 784-element vector
    flat = pixels.flatten()

    # Apply deskewing (same as training)
    flat = deskew(flat)

    return flat


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    image: str  # base64-encoded PNG image data


class PredictResponse(BaseModel):
    digit: int
    confidence: float
    probabilities: list[float]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    """API information."""
    return {
        "name": "Digit Recognition API",
        "version": "1.0.0",
        "model": "KNN (k=4, distance-weighted)",
        "endpoints": {
            "POST /predict": "Predict a handwritten digit from a base64 image",
            "GET /health": "Health check",
        },
    }


@app.get("/health")
def health():
    """Health check endpoint."""
    model_loaded = knn_model is not None and pca_model is not None
    return {
        "status": "healthy" if model_loaded else "model_not_loaded",
        "model_loaded": model_loaded,
        "metadata": model_metadata,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Predict the digit in a base64-encoded image from a canvas element.
    Returns the predicted digit, confidence score, and probability distribution.
    """
    if knn_model is None or pca_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run train_model.py first.",
        )

    try:
        # Decode base64 image
        # Handle both raw base64 and data-URL format (data:image/png;base64,...)
        image_data = request.image
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        image_bytes = base64.b64decode(image_data)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data.")

    try:
        # Preprocess the image
        features = preprocess_canvas_image(image_bytes)

        # Apply PCA transform (same as training)
        features_pca = pca_model.transform(features.reshape(1, -1))

        # Predict using KNN
        prediction = int(knn_model.predict(features_pca)[0])

        # Get distances and indices of nearest neighbors for confidence scoring
        distances, indices = knn_model.kneighbors(features_pca)

        # Compute probability-like scores for each digit (0-9)
        # Using inverse-distance weighting, same as the model's 'distance' weight
        neighbor_labels = knn_model._y[indices[0]]
        neighbor_distances = distances[0]

        # Avoid division by zero: if distance is 0, give it a very high weight
        weights = np.where(
            neighbor_distances < 1e-10,
            1e10,
            1.0 / neighbor_distances,
        )

        # Accumulate weights per digit class
        probabilities = np.zeros(10, dtype=np.float64)
        for label, weight in zip(neighbor_labels, weights):
            probabilities[int(label)] += weight

        # Normalize to sum to 1
        total_weight = probabilities.sum()
        if total_weight > 0:
            probabilities /= total_weight

        confidence = float(probabilities[prediction])

        return PredictResponse(
            digit=prediction,
            confidence=round(confidence, 4),
            probabilities=[round(float(p), 4) for p in probabilities],
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )
