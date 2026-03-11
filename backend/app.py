"""
FastAPI backend for handwritten digit recognition using a KNN model.
Serves predictions for base64-encoded canvas images.
"""

import os
import io
import base64
import pickle
from typing import Optional
import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter, shift
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
    version="1.0.1",
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
def center_by_mass(img_28x28):
    """
    Shift the image so its center of mass is at (14, 14).
    This matches how MNIST digits are centered.
    """
    total = img_28x28.sum()
    if total < 1e-10:
        return img_28x28

    rows, cols = np.mgrid[0:28, 0:28]
    cy = (rows * img_28x28).sum() / total
    cx = (cols * img_28x28).sum() / total

    shift_y = 14.0 - cy
    shift_x = 14.0 - cx

    shifted = shift(img_28x28, [shift_y, shift_x], order=1, mode='constant', cval=0.0)
    return shifted


def preprocess_canvas_image(image_bytes: bytes):
    """
    Convert a raw PNG image (from an HTML canvas) into a 784-element feature
    vector that matches the training preprocessing pipeline.

    Returns (features_flat, processed_28x28_image_array) so we can also
    return the processed image to the frontend for visualization.

    Steps:
      1. Open image, convert to grayscale
      2. Invert colors (canvas = white bg → MNIST = black bg)
      3. Find bounding box of the digit
      4. Crop and resize to 20x20 (maintaining aspect ratio)
      5. Place in 28x28 image
      6. Center by center of mass (matching MNIST exactly)
      7. Apply light Gaussian smoothing
      8. Normalize to [0, 1]
      9. Deskew
    """
    # Open and convert to grayscale
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    pixels = np.array(img, dtype=np.float64)

    # Invert: canvas has white bg (255) + black ink (0) → MNIST black bg + white ink
    pixels = 255.0 - pixels

    # Find bounding box of non-zero pixels (the digit)
    threshold = 30
    rows = np.any(pixels > threshold, axis=1)
    cols = np.any(pixels > threshold, axis=0)

    if not np.any(rows) or not np.any(cols):
        empty = np.zeros(784, dtype=np.float64)
        return empty, np.zeros((28, 28), dtype=np.float64)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Crop with padding
    pad = 20
    rmin = max(0, rmin - pad)
    rmax = min(pixels.shape[0] - 1, rmax + pad)
    cmin = max(0, cmin - pad)
    cmax = min(pixels.shape[1] - 1, cmax + pad)
    cropped = pixels[rmin:rmax + 1, cmin:cmax + 1]

    # Resize to fit in 20x20 maintaining aspect ratio
    crop_h, crop_w = cropped.shape
    if crop_h > crop_w:
        new_h = 20
        new_w = max(1, int(round(20.0 * crop_w / crop_h)))
    else:
        new_w = 20
        new_h = max(1, int(round(20.0 * crop_h / crop_w)))

    crop_img = Image.fromarray(cropped.astype(np.uint8))
    resized = crop_img.resize((new_w, new_h), Image.LANCZOS)
    resized_arr = np.array(resized, dtype=np.float64)

    # Place in 28x28 image
    result = np.zeros((28, 28), dtype=np.float64)
    top = (28 - new_h) // 2
    left = (28 - new_w) // 2
    result[top:top + new_h, left:left + new_w] = resized_arr

    # Center by center of mass (matching MNIST standard)
    result = center_by_mass(result)

    # Light Gaussian smoothing to match MNIST anti-aliasing style
    result = gaussian_filter(result, sigma=0.8)

    # Normalize to [0, 1]
    max_val = result.max()
    if max_val > 0:
        result = result / max_val

    # Save the processed image for visualization (before deskew)
    processed_img = (result * 255).astype(np.uint8)

    # Flatten and deskew
    flat = result.flatten()
    flat = deskew(flat)

    return flat, processed_img


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    image: str  # base64-encoded PNG image data


class PredictResponse(BaseModel):
    digit: int
    confidence: float
    probabilities: list[float]
    processed_image: Optional[str] = None  # base64 PNG of the 28x28 preprocessed image


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
        features, processed_img = preprocess_canvas_image(image_bytes)

        # Encode the processed 28x28 image as base64 PNG for frontend visualization
        proc_pil = Image.fromarray(processed_img)
        buf = io.BytesIO()
        proc_pil.save(buf, format="PNG")
        processed_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Apply PCA transform (same as training)
        features_pca = pca_model.transform(features.reshape(1, -1))

        # Predict using KNN
        prediction = int(knn_model.predict(features_pca)[0])

        # Get distances and indices of nearest neighbors for confidence scoring
        distances, indices = knn_model.kneighbors(features_pca)

        # Compute probability-like scores for each digit (0-9)
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
            processed_image=processed_b64,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )
