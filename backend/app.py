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
from scipy.ndimage import affine_transform, shift
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
def preprocess_canvas_image(image_bytes: bytes):
    """
    Convert a raw PNG image (from an HTML canvas) into a 784-element feature
    vector that matches the training preprocessing pipeline.

    Returns (features_flat, processed_28x28_image_array).

    Approach: Create a square crop centered on the digit's center of mass,
    sized so the digit fills ~71% (matching MNIST's 20/28 ratio).
    Resize that square to 28x28. This naturally handles any digit size.

    Steps:
      1. Open image, convert to grayscale, invert
      2. Find digit bounding box and center of mass
      3. Create square crop centered on center of mass (sized proportionally)
      4. Resize square to 28x28
      5. Shift so center of mass is at (14,14) — matching MNIST
      6. Normalize to [0, 1] (matching training: X / 255.0)
      7. Deskew (matching training)
    """
    # Open and convert to grayscale
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    pixels = np.array(img, dtype=np.float64)

    # Invert: canvas white bg (255) + black ink (0) → MNIST black bg + white ink
    pixels = 255.0 - pixels

    # Find bounding box of the digit
    threshold = 30
    row_mask = np.any(pixels > threshold, axis=1)
    col_mask = np.any(pixels > threshold, axis=0)

    if not np.any(row_mask) or not np.any(col_mask):
        empty = np.zeros(784, dtype=np.float64)
        return empty, np.zeros((28, 28), dtype=np.uint8)

    rmin, rmax = np.where(row_mask)[0][[0, -1]]
    cmin, cmax = np.where(col_mask)[0][[0, -1]]

    # Compute center of mass of the digit
    digit_region = pixels.copy()
    digit_region[digit_region < threshold] = 0
    total = digit_region.sum()
    rows_idx = np.arange(pixels.shape[0])[:, None]
    cols_idx = np.arange(pixels.shape[1])[None, :]
    cm_r = (rows_idx * digit_region).sum() / total
    cm_c = (cols_idx * digit_region).sum() / total

    # Determine square crop size
    # MNIST format: digit fits in ~20x20 inside 28x28 → digit is 71.4% of image
    # So our crop should be: digit_size / 0.714
    bbox_h = rmax - rmin + 1
    bbox_w = cmax - cmin + 1
    digit_size = max(bbox_h, bbox_w)
    crop_side = max(int(digit_size / 0.70), digit_size + 16)  # at least 8px padding each side

    # Create square crop centered on center of mass
    half = crop_side / 2.0
    r1 = int(cm_r - half)
    r2 = int(cm_r + half)
    c1 = int(cm_c - half)
    c2 = int(cm_c + half)

    # Extract with zero-padding if crop goes outside image bounds
    crop_h = r2 - r1
    crop_w = c2 - c1
    cropped = np.zeros((crop_h, crop_w), dtype=np.float64)

    # Compute overlap between crop and actual image
    src_r1 = max(0, r1)
    src_r2 = min(pixels.shape[0], r2)
    src_c1 = max(0, c1)
    src_c2 = min(pixels.shape[1], c2)
    dst_r1 = src_r1 - r1
    dst_r2 = dst_r1 + (src_r2 - src_r1)
    dst_c1 = src_c1 - c1
    dst_c2 = dst_c1 + (src_c2 - src_c1)
    cropped[dst_r1:dst_r2, dst_c1:dst_c2] = pixels[src_r1:src_r2, src_c1:src_c2]

    # Resize the square crop directly to 28x28
    crop_img = Image.fromarray(cropped.astype(np.uint8))
    resized_28 = crop_img.resize((28, 28), Image.LANCZOS)
    result = np.array(resized_28, dtype=np.float64)

    # Fine-tune: shift so center of mass is at (14, 14) — matching MNIST exactly
    total_28 = result.sum()
    if total_28 > 1e-10:
        r_idx, c_idx = np.mgrid[0:28, 0:28]
        cm_r_28 = (r_idx * result).sum() / total_28
        cm_c_28 = (c_idx * result).sum() / total_28
        shift_r = 14.0 - cm_r_28
        shift_c = 14.0 - cm_c_28
        if abs(shift_r) > 0.5 or abs(shift_c) > 0.5:
            result = shift(result, [shift_r, shift_c], order=1, mode='constant', cval=0.0)

    # Normalize to [0, 1] — must match training exactly (training uses X / 255.0)
    result = result / 255.0

    # Save processed image for visualization (before deskew)
    processed_img = (result * 255).astype(np.uint8)

    # Flatten and deskew (matching training pipeline)
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
