# Digit Recognition AI

> Handwritten Digit Recognition using K-Nearest Neighbors (KNN) — achieving **98.47% accuracy** on MNIST

[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen?style=for-the-badge)](https://skullfire07.github.io/ML-FINAL/)
[![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-KNN-F7931E?style=for-the-badge&logo=scikitlearn)](https://scikit-learn.org)

---

## Overview

A full-stack ML application that recognizes handwritten digits in real-time. Draw a digit on the canvas and watch the AI classify it instantly with confidence scores for all 10 digits.

### Key Features

- **Draw & Predict** — Interactive canvas with smooth Bezier curve drawing + touch support
- **98.47% Accuracy** — KNN with deskewing preprocessing + PCA dimensionality reduction
- **Beautiful UI** — Glassmorphism design, animated probability charts, particle background, dark theme
- **Full-Stack** — FastAPI backend + vanilla JS frontend, deployed on Railway + GitHub Pages
- **Comprehensive Analysis** — Confusion matrix, K-optimization, PCA variance, misclassified examples

---

## Model Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **98.47%** |
| Error Rate | 1.53% |
| Best K | 3 |
| Distance Metric | Minkowski (Euclidean) |
| Voting | Distance-weighted |
| PCA Components | 50 (87.1% variance retained) |
| Preprocessing | Deskewing (moment-based affine transform) |
| Cross-Validation | 97.93% ± 0.26% (5-fold) |
| Training Samples | 60,000 |
| Test Samples | 10,000 |

### Per-Class Accuracy

| Digit | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|-------|------|------|------|------|------|------|------|------|------|------|
| **Accuracy** | 99.5% | 99.5% | 98.5% | 97.6% | 97.9% | 98.3% | 99.5% | 98.5% | 97.0% | 98.2% |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Model | K-Nearest Neighbors (scikit-learn) |
| Preprocessing | Deskewing + PCA (50 components) |
| Dataset | MNIST (70,000 handwritten digits, 28×28 px) |
| Backend | Python, FastAPI, Uvicorn |
| Frontend | HTML5 Canvas, CSS3 (Glassmorphism), Vanilla JS |
| Backend Hosting | Railway |
| Frontend Hosting | GitHub Pages |

---

## How It Works

```
Draw Digit → Preprocess (resize, grayscale, invert, deskew) → PCA Transform → KNN Classify → Result
```

1. **Draw** a digit (0-9) on the interactive canvas
2. **Preprocess** — Image is resized to 28×28, converted to grayscale, colors inverted to match MNIST format, and deskewed using moment-based affine transformation
3. **PCA** — Reduced from 784 dimensions to 50 principal components (87.1% variance retained)
4. **KNN** — K=3, distance-weighted Minkowski voting finds the nearest training examples
5. **Result** — Predicted digit with confidence scores and probability distribution for all 10 classes

### Why Deskewing?

Deskewing straightens slanted digit images by computing second-order central moments and applying an affine shear correction. This single preprocessing step **cuts the error rate nearly in half** (from ~3.1% to ~1.69%).

### Why KNN?

While deep learning (CNNs) can achieve 99.7%+ on MNIST, KNN offers:
- **Interpretability** — predictions based on actual similar training examples
- **No training phase** — lazy learner, instant model updates
- **Simplicity** — minimal hyperparameter tuning
- **Competitive accuracy** — 98.47% with proper preprocessing

---

## Project Structure

```
ML-FINAL/
├── frontend/                    # GitHub Pages frontend
│   ├── index.html               # Single-page app (HTML + CSS + JS)
│   ├── 404.html                 # SPA redirect for GitHub Pages
│   └── CNAME                    # Custom domain (optional)
├── backend/                     # Railway backend
│   ├── app.py                   # FastAPI application (/predict, /health)
│   ├── train_model.py           # Model training script
│   ├── model/                   # Saved model artifacts
│   │   ├── knn_model.pkl        # Trained KNN classifier
│   │   ├── pca.pkl              # Fitted PCA transformer
│   │   └── metadata.pkl         # Model metadata & metrics
│   ├── requirements.txt         # Python dependencies
│   ├── Procfile                 # Railway process file
│   └── railway.json             # Railway deployment config
├── knn_digit_recognition.py     # Standalone ML pipeline with 6 visualizations
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/SKULLFIRE07/ML-FINAL.git
cd ML-FINAL
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r backend/requirements.txt
pip install matplotlib seaborn  # For visualization script
```

### 2. Train the Model (optional — pre-trained model included)

```bash
cd backend
python train_model.py
```

### 3. Run the Backend

```bash
cd backend
uvicorn app:app --reload --port 8000
```

### 4. Open the Frontend

Update `API_URL` in `frontend/index.html` to `http://localhost:8000`, then:

```bash
cd frontend
python -m http.server 3000
```

Open http://localhost:3000 and start drawing!

### 5. Run Full Visualization Pipeline (optional)

```bash
python knn_digit_recognition.py
```

Generates 6 plots: sample digits, deskew comparison, PCA variance, K vs accuracy, confusion matrix, misclassified examples.

---

## Deployment

### Frontend → GitHub Pages

1. Push to GitHub
2. Go to **Settings → Pages → Source**: `main` branch, `/frontend` folder
3. App is live at `https://skullfire07.github.io/ML-FINAL/`

### Backend → Railway

1. Connect your GitHub repo to [Railway](https://railway.app)
2. Set root directory to `/backend`
3. Railway auto-detects Python and deploys
4. Update `API_URL` in `frontend/index.html` to your Railway URL

---

## API Reference

### `POST /predict`

Predict a handwritten digit from a base64-encoded image.

**Request:**
```json
{
  "image": "data:image/png;base64,iVBORw0KGgo..."
}
```

**Response:**
```json
{
  "digit": 7,
  "confidence": 0.9832,
  "probabilities": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9832, 0.0168, 0.0]
}
```

### `GET /health`

Returns model status and metadata.

### `GET /`

Returns API info and available endpoints.

---

## References

- [MNIST Database](http://yann.lecun.com/exdb/mnist/) — Yann LeCun et al.
- [Shape Context Matching](https://www.researchgate.net/publication/3853568) — Belongie, Malik, Puzicha (2002) — 0.63% error with 1-NN
- [Modified Sliding-Window Metric for KNN on MNIST](https://arxiv.org/abs/1809.06846)
- [MNIST Perfect 100% using kNN](https://www.kaggle.com/code/cdeotte/mnist-perfect-100-using-knn) — Chris Deotte, Kaggle

---

Built with ML by [SKULLFIRE07](https://github.com/SKULLFIRE07)
