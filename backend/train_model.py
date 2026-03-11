"""
Train a KNN model on the MNIST dataset with deskewing preprocessing and PCA.
Saves the trained model, PCA transformer, and preprocessing artifacts to backend/model/.
"""

import os
import pickle
import time
import numpy as np
from scipy.ndimage import affine_transform
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def deskew(image):
    """
    Deskew a single 28x28 image using moment-based affine transformation.
    Straightens slanted digits by computing the second-order central moments
    and applying a shear correction.
    """
    img = image.reshape(28, 28)
    # Compute moments
    m = np.float64(img)
    total = m.sum()
    if total < 1e-10:
        return image  # Return as-is if the image is blank

    # Center of mass
    grid_x, grid_y = np.mgrid[0:28, 0:28]
    cx = (grid_x * m).sum() / total
    cy = (grid_y * m).sum() / total

    # Second-order central moments
    mu20 = (grid_x * grid_x * m).sum() / total - cx * cx
    mu02 = (grid_y * grid_y * m).sum() / total - cy * cy
    mu11 = (grid_x * grid_y * m).sum() / total - cx * cy

    # Compute skew
    if mu02 < 1e-10:
        return image
    skew = mu11 / mu02

    # Affine transformation matrix to deskew
    transform = np.array([
        [1, 0],
        [skew, 1]
    ])

    # Compute the offset so that the center of mass stays centered
    offset = np.array([cx, cy]) - np.dot(transform, [14.0, 14.0])

    # Apply affine transformation
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


def deskew_dataset(X):
    """Apply deskewing to an entire dataset."""
    print("Deskewing images...")
    X_deskewed = np.zeros_like(X)
    for i in range(len(X)):
        X_deskewed[i] = deskew(X[i])
        if (i + 1) % 10000 == 0:
            print(f"  Deskewed {i + 1}/{len(X)} images")
    print(f"  Deskewed all {len(X)} images")
    return X_deskewed


def main():
    start_time = time.time()

    # Create model directory
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
    os.makedirs(model_dir, exist_ok=True)

    # Load MNIST dataset
    print("Loading MNIST dataset...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    X, y = mnist.data.astype(np.float64), mnist.target.astype(np.int32)
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

    # Normalize pixel values to [0, 1]
    # NOTE: We do NOT use StandardScaler — it hurts KNN performance on MNIST
    # because it distorts the pixel value distances that KNN relies on.
    X = X / 255.0

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Apply deskewing
    X_train_deskewed = deskew_dataset(X_train)
    X_test_deskewed = deskew_dataset(X_test)

    # Apply PCA for dimensionality reduction
    print("Fitting PCA with 50 components...")
    pca = PCA(n_components=50, random_state=42)
    X_train_pca = pca.fit_transform(X_train_deskewed)
    X_test_pca = pca.transform(X_test_deskewed)
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"PCA explained variance: {explained_var:.4f} ({explained_var * 100:.2f}%)")

    # Train KNN classifier
    print("Training KNN classifier (k=4, distance-weighted, Minkowski p=2)...")
    knn = KNeighborsClassifier(
        n_neighbors=4,
        weights='distance',
        metric='minkowski',
        p=2,
        n_jobs=-1
    )
    knn.fit(X_train_pca, y_train)
    print("Training complete.")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    y_pred = knn.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model artifacts
    print("\nSaving model artifacts...")

    with open(os.path.join(model_dir, "knn_model.pkl"), "wb") as f:
        pickle.dump(knn, f)
    print("  Saved knn_model.pkl")

    with open(os.path.join(model_dir, "pca.pkl"), "wb") as f:
        pickle.dump(pca, f)
    print("  Saved pca.pkl")

    # Save metadata for reference
    metadata = {
        "n_neighbors": 4,
        "weights": "distance",
        "metric": "minkowski",
        "p": 2,
        "pca_components": 50,
        "pca_explained_variance": float(explained_var),
        "test_accuracy": float(accuracy),
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "preprocessing": ["normalize_0_1", "deskew", "pca_50"],
    }
    with open(os.path.join(model_dir, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)
    print("  Saved metadata.pkl")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f} seconds")
    print("All model artifacts saved to backend/model/")


if __name__ == "__main__":
    main()
