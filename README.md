# Digit Recognition AI

> Handwritten Digit Recognition using K-Nearest Neighbors (KNN) — achieving **98%+ accuracy** on MNIST

[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen?style=for-the-badge)](https://your-username.github.io/ML-FINAL/)
[![Backend API](https://img.shields.io/badge/API-Railway-blueviolet?style=for-the-badge)](https://your-backend.railway.app)

## Overview

A full-stack ML application that recognizes handwritten digits in real-time. Draw a digit on the canvas and watch the AI classify it instantly with confidence scores.

### Key Features

- **Draw & Predict** — Interactive canvas with smooth drawing, instant predictions
- **98%+ Accuracy** — KNN with deskewing preprocessing + PCA dimensionality reduction
- **Beautiful UI** — Glassmorphism design, animated probability charts, dark theme
- **Full-Stack** — FastAPI backend, vanilla JS frontend, deployed on Railway + GitHub Pages

## Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Model | K-Nearest Neighbors (scikit-learn) |
| Preprocessing | Deskewing + PCA (50 components) |
| Dataset | MNIST (70,000 handwritten digits) |
| Backend | FastAPI + Uvicorn |
| Frontend | HTML5 Canvas + CSS3 + Vanilla JS |
| Backend Hosting | Railway |
| Frontend Hosting | GitHub Pages |

## How It Works

```
Draw Digit → Preprocess (resize, grayscale, deskew) → PCA Transform → KNN Classify → Result
```

1. **Draw** a digit (0-9) on the canvas
2. **Preprocess** — Image is resized to 28×28, converted to grayscale, and deskewed
3. **PCA** — Reduced from 784 dimensions to 50 principal components
4. **KNN** — K=4, distance-weighted voting finds the nearest training examples
5. **Result** — Predicted digit with confidence scores for all 10 classes

## Project Structure

```
ML-FINAL/
├── frontend/           # GitHub Pages frontend
│   └── index.html      # Single-page app (HTML + CSS + JS)
├── backend/            # Railway backend
│   ├── app.py          # FastAPI application
│   ├── train_model.py  # Model training script
│   ├── model/          # Saved model files (generated)
│   ├── requirements.txt
│   ├── Procfile         # Railway process file
│   └── railway.json     # Railway config
├── knn_digit_recognition.py  # Standalone ML pipeline with visualizations
└── README.md
```

## Quick Start

### 1. Train the Model

```bash
cd backend
pip install -r requirements.txt
python train_model.py
```

### 2. Run the Backend

```bash
cd backend
uvicorn app:app --reload --port 8000
```

### 3. Open the Frontend

Open `frontend/index.html` in your browser, or serve it:

```bash
cd frontend
python -m http.server 3000
```

Update the `API_URL` in `index.html` to point to your backend (`http://localhost:8000` for local dev).

## Deployment

### Frontend → GitHub Pages

1. Push to GitHub
2. Go to Settings → Pages → Source: `main` branch, `/frontend` folder
3. Your app is live at `https://username.github.io/repo-name/`

### Backend → Railway

1. Connect your GitHub repo to Railway
2. Set root directory to `/backend`
3. Railway auto-detects Python and deploys
4. Update `API_URL` in frontend to your Railway URL

## Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | **~98%+** |
| K value | 4 |
| Distance metric | Minkowski (Euclidean) |
| Voting | Distance-weighted |
| PCA components | 50 |
| Preprocessing | Deskewing |

## Why KNN?

While deep learning (CNNs) can achieve 99.7%+ on MNIST, KNN offers:
- **Interpretability** — predictions based on actual similar training examples
- **No training time** — lazy learner, instant model updates
- **Simplicity** — no hyperparameter tuning nightmare
- **Competitive accuracy** — 98%+ with proper preprocessing

---

Built with ML | [GitHub](https://github.com/your-username/ML-FINAL)
