"""
Generate a comprehensive PDF report for the Digit Recognition AI project.
Uses matplotlib PdfPages to create a multi-page report.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
import os

BASE = "/home/aryan-budukh/Desktop/ML-FINAL"
OUTPUT = os.path.join(BASE, "Digit_Recognition_Project_Report.pdf")

# Colors
DARK = "#1a1a2e"
ACCENT = "#e94560"
BLUE = "#0f3460"
LIGHT = "#f5f5f5"
WHITE = "#ffffff"


def new_page(fig_size=(8.5, 11)):
    fig = plt.figure(figsize=fig_size)
    fig.patch.set_facecolor(WHITE)
    return fig


def add_footer(fig, page_num, total=14):
    fig.text(0.5, 0.02, f"Page {page_num} of {total}",
             ha="center", fontsize=8, color="gray")


def page_title_page():
    fig = new_page()
    # Background
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Top accent bar
    ax.axhspan(0.88, 1.0, color=DARK)
    # Bottom accent bar
    ax.axhspan(0, 0.08, color=DARK)
    # Middle accent line
    ax.axhline(y=0.52, xmin=0.2, xmax=0.8, color=ACCENT, linewidth=3)

    ax.text(0.5, 0.75, "Handwritten Digit Recognition",
            ha="center", va="center", fontsize=28, fontweight="bold", color=DARK)
    ax.text(0.5, 0.67, "using K-Nearest Neighbors",
            ha="center", va="center", fontsize=24, fontweight="bold", color=BLUE)
    ax.text(0.5, 0.58, "Machine Learning Project Report",
            ha="center", va="center", fontsize=16, color="gray")

    ax.text(0.5, 0.42, "Author:  Aryan Budukh",
            ha="center", va="center", fontsize=14, color=DARK)
    ax.text(0.5, 0.36, "Date:  March 2026",
            ha="center", va="center", fontsize=14, color=DARK)

    ax.text(0.5, 0.22, "MNIST Dataset  |  scikit-learn  |  FastAPI  |  GitHub Pages",
            ha="center", va="center", fontsize=11, color="gray")

    add_footer(fig, 1)
    return fig


def page_overview():
    fig = new_page()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.92, "Project Overview", ha="center", va="center",
            fontsize=24, fontweight="bold", color=DARK)
    ax.axhline(y=0.895, xmin=0.25, xmax=0.75, color=ACCENT, linewidth=2)

    sections = [
        ("Objective", [
            "Build a handwritten digit recognition system using K-Nearest Neighbors (KNN)",
            "on the MNIST dataset, with a web-based interactive demo."
        ]),
        ("Dataset: MNIST", [
            "70,000 grayscale images of handwritten digits (0-9), each 28x28 pixels.",
            "Training set: 60,000 samples  |  Test set: 10,000 samples.",
            "One of the most well-studied benchmarks in machine learning."
        ]),
        ("Approach", [
            "1. Deskewing preprocessing: moment-based affine transformation.",
            "2. PCA dimensionality reduction: 784 -> 50 principal components.",
            "3. Distance-weighted KNN classification with K=3."
        ]),
        ("Tech Stack", [
            "Python, scikit-learn, NumPy, matplotlib, FastAPI,",
            "HTML5 Canvas, CSS3, JavaScript, GitHub Pages, Railway."
        ]),
    ]

    y = 0.84
    for title, lines in sections:
        ax.text(0.1, y, title, fontsize=14, fontweight="bold", color=BLUE)
        y -= 0.03
        for line in lines:
            ax.text(0.12, y, line, fontsize=10, color=DARK)
            y -= 0.025
        y -= 0.025

    add_footer(fig, 2)
    return fig


def page_preprocessing():
    fig = new_page()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.92, "Preprocessing Pipeline", ha="center", va="center",
            fontsize=24, fontweight="bold", color=DARK)
    ax.axhline(y=0.895, xmin=0.25, xmax=0.75, color=ACCENT, linewidth=2)

    y = 0.84
    ax.text(0.1, y, "1. Deskewing", fontsize=14, fontweight="bold", color=BLUE)
    y -= 0.03
    deskew_lines = [
        "Moment-based affine transformation to straighten slanted digits.",
        "Computes the second-order central moments of the image to determine",
        "the skew direction, then applies an affine warp to correct it.",
        "Impact: Reduces error rate from ~3.1% to ~1.69% (a 45% reduction).",
    ]
    for line in deskew_lines:
        ax.text(0.12, y, line, fontsize=10, color=DARK)
        y -= 0.025

    y -= 0.02
    ax.text(0.1, y, "2. PCA Dimensionality Reduction", fontsize=14, fontweight="bold", color=BLUE)
    y -= 0.03
    pca_lines = [
        "Reduces 784-dimensional pixel space to 50 principal components.",
        "Retains 87.1% of the total variance.",
        "Benefits: Faster KNN inference, reduced noise, better generalization.",
    ]
    for line in pca_lines:
        ax.text(0.12, y, line, fontsize=10, color=DARK)
        y -= 0.025

    # Load PCA variance image
    y -= 0.02
    img_path = os.path.join(BASE, "pca_variance.png")
    img = mpimg.imread(img_path)
    img_ax = fig.add_axes([0.12, 0.08, 0.76, 0.42])
    img_ax.imshow(img)
    img_ax.axis("off")
    img_ax.set_title("PCA Explained Variance vs Number of Components", fontsize=10, color=DARK)

    add_footer(fig, 3)
    return fig


def page_sample_digits():
    fig = new_page()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.94, "Sample Digits", ha="center", va="center",
            fontsize=24, fontweight="bold", color=DARK)
    ax.axhline(y=0.915, xmin=0.25, xmax=0.75, color=ACCENT, linewidth=2)

    img_path = os.path.join(BASE, "sample_digits.png")
    img = mpimg.imread(img_path)
    img_ax = fig.add_axes([0.05, 0.12, 0.9, 0.75])
    img_ax.imshow(img)
    img_ax.axis("off")

    ax.text(0.5, 0.08, "Sample digits from MNIST dataset -- 10 examples per class",
            ha="center", va="center", fontsize=11, style="italic", color="gray")

    add_footer(fig, 4)
    return fig


def page_deskew_comparison():
    fig = new_page()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.94, "Deskewing Comparison", ha="center", va="center",
            fontsize=24, fontweight="bold", color=DARK)
    ax.axhline(y=0.915, xmin=0.25, xmax=0.75, color=ACCENT, linewidth=2)

    img_path = os.path.join(BASE, "deskew_comparison.png")
    img = mpimg.imread(img_path)
    img_ax = fig.add_axes([0.05, 0.15, 0.9, 0.72])
    img_ax.imshow(img)
    img_ax.axis("off")

    ax.text(0.5, 0.09, "Effect of deskewing preprocessing -- before (top) vs after (bottom)",
            ha="center", va="center", fontsize=11, style="italic", color="gray")

    add_footer(fig, 5)
    return fig


def page_hyperparameter():
    fig = new_page()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.94, "Hyperparameter Optimization", ha="center", va="center",
            fontsize=24, fontweight="bold", color=DARK)
    ax.axhline(y=0.915, xmin=0.25, xmax=0.75, color=ACCENT, linewidth=2)

    # K vs accuracy image
    img_path = os.path.join(BASE, "k_vs_accuracy.png")
    img = mpimg.imread(img_path)
    img_ax = fig.add_axes([0.1, 0.42, 0.8, 0.46])
    img_ax.imshow(img)
    img_ax.axis("off")

    # Table
    k_data = [
        ["K", "Accuracy", ""],
        ["1", "98.28%", ""],
        ["2", "98.28%", ""],
        ["3", "98.47%", "BEST"],
        ["4", "98.44%", ""],
        ["5", "98.37%", ""],
        ["7", "98.35%", ""],
        ["9", "98.35%", ""],
        ["11", "98.34%", ""],
    ]

    table_ax = fig.add_axes([0.2, 0.08, 0.6, 0.30])
    table_ax.axis("off")
    table = table_ax.table(
        cellText=[[r[0], r[1], r[2]] for r in k_data],
        colLabels=None,
        colWidths=[0.25, 0.4, 0.35],
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Header row styling
    for j in range(3):
        table[0, j].set_facecolor(DARK)
        table[0, j].set_text_props(color=WHITE, fontweight="bold")

    # Highlight best row (K=3, row index 3)
    for j in range(3):
        table[3, j].set_facecolor("#d4edda")

    add_footer(fig, 6)
    return fig


def page_model_config():
    fig = new_page()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.92, "Model Configuration", ha="center", va="center",
            fontsize=24, fontweight="bold", color=DARK)
    ax.axhline(y=0.895, xmin=0.25, xmax=0.75, color=ACCENT, linewidth=2)

    config_data = [
        ["Parameter", "Value"],
        ["Algorithm", "K-Nearest Neighbors"],
        ["K", "3 (optimal)"],
        ["Distance Metric", "Minkowski (Euclidean, p=2)"],
        ["Weights", "Distance-weighted"],
        ["PCA Components", "50"],
        ["Preprocessing", "Deskewing + PCA"],
        ["Cross-Validation", "5-fold, 97.93% +/- 0.26%"],
    ]

    table_ax = fig.add_axes([0.1, 0.35, 0.8, 0.48])
    table_ax.axis("off")
    table = table_ax.table(
        cellText=config_data,
        colLabels=None,
        colWidths=[0.45, 0.55],
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.2)

    for j in range(2):
        table[0, j].set_facecolor(DARK)
        table[0, j].set_text_props(color=WHITE, fontweight="bold", fontsize=12)

    for i in range(1, len(config_data)):
        table[i, 0].set_text_props(fontweight="bold")
        if i % 2 == 0:
            for j in range(2):
                table[i, j].set_facecolor("#f0f0f0")

    ax.text(0.5, 0.25, "Distance-weighted KNN gives closer neighbors more influence,",
            ha="center", fontsize=11, color=DARK)
    ax.text(0.5, 0.22, "which improves accuracy compared to uniform weighting.",
            ha="center", fontsize=11, color=DARK)

    add_footer(fig, 7)
    return fig


def page_results():
    fig = new_page()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.94, "Results", ha="center", va="center",
            fontsize=24, fontweight="bold", color=DARK)
    ax.axhline(y=0.915, xmin=0.25, xmax=0.75, color=ACCENT, linewidth=2)

    # Big accuracy number
    ax.text(0.5, 0.84, "98.47%", ha="center", va="center",
            fontsize=64, fontweight="bold", color=ACCENT)
    ax.text(0.5, 0.78, "Test Accuracy", ha="center", va="center",
            fontsize=18, color=DARK)

    ax.text(0.5, 0.73, "Error Rate: 1.53%   |   153 misclassified out of 10,000 test samples",
            ha="center", va="center", fontsize=11, color="gray")

    # Confusion matrix image
    img_path = os.path.join(BASE, "confusion_matrix.png")
    img = mpimg.imread(img_path)
    img_ax = fig.add_axes([0.08, 0.06, 0.84, 0.62])
    img_ax.imshow(img)
    img_ax.axis("off")

    add_footer(fig, 8)
    return fig


def page_per_class():
    fig = new_page()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.92, "Per-Class Performance", ha="center", va="center",
            fontsize=24, fontweight="bold", color=DARK)
    ax.axhline(y=0.895, xmin=0.25, xmax=0.75, color=ACCENT, linewidth=2)

    header = ["Digit", "Precision", "Recall", "F1-Score", "Support"]
    data = [
        ["0", "0.9849", "0.9949", "0.9899", "986"],
        ["1", "0.9894", "0.9947", "0.9920", "1125"],
        ["2", "0.9919", "0.9850", "0.9884", "999"],
        ["3", "0.9871", "0.9765", "0.9818", "1020"],
        ["4", "0.9866", "0.9795", "0.9830", "975"],
        ["5", "0.9779", "0.9834", "0.9807", "902"],
        ["6", "0.9889", "0.9949", "0.9919", "982"],
        ["7", "0.9818", "0.9846", "0.9832", "1042"],
        ["8", "0.9885", "0.9703", "0.9793", "975"],
        ["9", "0.9692", "0.9819", "0.9755", "994"],
    ]

    table_ax = fig.add_axes([0.08, 0.22, 0.84, 0.62])
    table_ax.axis("off")

    table = table_ax.table(
        cellText=data,
        colLabels=header,
        colWidths=[0.12, 0.22, 0.22, 0.22, 0.22],
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    for j in range(5):
        table[0, j].set_facecolor(DARK)
        table[0, j].set_text_props(color=WHITE, fontweight="bold")

    for i in range(1, 11):
        if i % 2 == 0:
            for j in range(5):
                table[i, j].set_facecolor("#f0f0f0")

    # Highlight best and worst
    ax.text(0.5, 0.16, "Best F1-Score: Digit 1 (0.9920)  |  Worst F1-Score: Digit 9 (0.9755)",
            ha="center", fontsize=11, color=DARK)
    ax.text(0.5, 0.12, "All classes achieve F1-Score above 0.97 -- strong balanced performance.",
            ha="center", fontsize=11, color="gray")

    add_footer(fig, 9)
    return fig


def page_misclassified():
    fig = new_page()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.94, "Misclassified Examples", ha="center", va="center",
            fontsize=24, fontweight="bold", color=DARK)
    ax.axhline(y=0.915, xmin=0.25, xmax=0.75, color=ACCENT, linewidth=2)

    img_path = os.path.join(BASE, "misclassified.png")
    img = mpimg.imread(img_path)
    img_ax = fig.add_axes([0.05, 0.22, 0.9, 0.65])
    img_ax.imshow(img)
    img_ax.axis("off")

    ax.text(0.5, 0.16, "Analysis of Misclassifications", ha="center",
            fontsize=13, fontweight="bold", color=BLUE)
    ax.text(0.5, 0.12,
            "Most errors occur between visually similar digits: 4<->9, 3<->8, 7<->9.",
            ha="center", fontsize=11, color=DARK)
    ax.text(0.5, 0.08,
            "These are cases where even humans might struggle with ambiguous handwriting.",
            ha="center", fontsize=10, color="gray")

    add_footer(fig, 10)
    return fig


def page_architecture():
    fig = new_page()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.92, "System Architecture", ha="center", va="center",
            fontsize=24, fontweight="bold", color=DARK)
    ax.axhline(y=0.895, xmin=0.25, xmax=0.75, color=ACCENT, linewidth=2)

    # Draw architecture diagram using patches and text
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    # Frontend box
    frontend_box = FancyBboxPatch((0.05, 0.55), 0.38, 0.28,
                                   boxstyle="round,pad=0.02",
                                   facecolor="#e8f4f8", edgecolor=BLUE, linewidth=2)
    ax.add_patch(frontend_box)
    ax.text(0.24, 0.80, "Frontend", ha="center", fontsize=14, fontweight="bold", color=BLUE)
    ax.text(0.24, 0.76, "(GitHub Pages)", ha="center", fontsize=10, color="gray")
    ax.text(0.24, 0.70, "HTML5 Canvas", ha="center", fontsize=10, color=DARK)
    ax.text(0.24, 0.66, "CSS3 + JavaScript", ha="center", fontsize=10, color=DARK)
    ax.text(0.24, 0.62, "Draw digit interface", ha="center", fontsize=10, color=DARK)

    # Backend box
    backend_box = FancyBboxPatch((0.57, 0.55), 0.38, 0.28,
                                  boxstyle="round,pad=0.02",
                                  facecolor="#fff3e0", edgecolor=ACCENT, linewidth=2)
    ax.add_patch(backend_box)
    ax.text(0.76, 0.80, "Backend", ha="center", fontsize=14, fontweight="bold", color=ACCENT)
    ax.text(0.76, 0.76, "(Railway)", ha="center", fontsize=10, color="gray")
    ax.text(0.76, 0.70, "FastAPI Server", ha="center", fontsize=10, color=DARK)
    ax.text(0.76, 0.66, "Deskew + PCA", ha="center", fontsize=10, color=DARK)
    ax.text(0.76, 0.62, "KNN Predict", ha="center", fontsize=10, color=DARK)

    # Arrows
    ax.annotate("", xy=(0.57, 0.73), xytext=(0.43, 0.73),
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=2))
    ax.text(0.50, 0.745, "base64 image", ha="center", fontsize=8, color=BLUE)

    ax.annotate("", xy=(0.43, 0.63), xytext=(0.57, 0.63),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=2))
    ax.text(0.50, 0.615, "JSON response", ha="center", fontsize=8, color=ACCENT)

    # Pipeline
    ax.text(0.5, 0.46, "Request Pipeline", ha="center", fontsize=14,
            fontweight="bold", color=DARK)

    steps = [
        "1. User draws digit on HTML5 Canvas",
        "2. Canvas image encoded as base64 PNG",
        "3. POST request to FastAPI /predict endpoint",
        "4. Backend: decode image -> deskew -> PCA transform",
        "5. KNN classifies the preprocessed image",
        "6. Predicted digit + confidence returned as JSON",
        "7. Frontend displays result with confidence bar",
    ]
    y = 0.40
    for step in steps:
        ax.text(0.15, y, step, fontsize=10, color=DARK)
        y -= 0.03

    add_footer(fig, 11)
    return fig


def page_deployment():
    fig = new_page()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.92, "Deployment", ha="center", va="center",
            fontsize=24, fontweight="bold", color=DARK)
    ax.axhline(y=0.895, xmin=0.25, xmax=0.75, color=ACCENT, linewidth=2)

    y = 0.82
    sections = [
        ("Frontend: GitHub Pages", [
            "URL: https://skullfire07.github.io/ML-FINAL/",
            "Static site served via GitHub Pages.",
            "HTML5 Canvas for digit drawing, JavaScript for API calls.",
        ]),
        ("Backend: Railway", [
            "URL: https://digit-recognition-production.up.railway.app",
            "FastAPI application with automatic deployment from GitHub.",
            "Includes /predict endpoint for digit classification.",
            "Model artifacts (KNN, PCA, scaler) loaded at startup.",
        ]),
        ("Source Code: GitHub", [
            "Repository: https://github.com/SKULLFIRE07/ML-FINAL",
            "Includes training scripts, API server, frontend, and documentation.",
        ]),
        ("Key Files", [
            "train_model.py -- Model training and evaluation pipeline",
            "main.py -- FastAPI backend server",
            "index.html -- Frontend web application",
            "knn_model.pkl -- Trained KNN model artifact",
            "pca_model.pkl -- Fitted PCA transformer",
        ]),
    ]

    for title, lines in sections:
        ax.text(0.1, y, title, fontsize=14, fontweight="bold", color=BLUE)
        y -= 0.035
        for line in lines:
            ax.text(0.12, y, line, fontsize=10, color=DARK)
            y -= 0.028
        y -= 0.025

    add_footer(fig, 12)
    return fig


def page_conclusion():
    fig = new_page()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.92, "Conclusion", ha="center", va="center",
            fontsize=24, fontweight="bold", color=DARK)
    ax.axhline(y=0.895, xmin=0.25, xmax=0.75, color=ACCENT, linewidth=2)

    y = 0.82
    ax.text(0.1, y, "Key Achievements", fontsize=14, fontweight="bold", color=BLUE)
    y -= 0.04
    achievements = [
        "Achieved 98.47% accuracy using KNN on MNIST -- competitive with simple neural networks.",
        "Built a full end-to-end pipeline: training, evaluation, API, and web frontend.",
        "Deployed as a live interactive demo accessible via browser.",
    ]
    for line in achievements:
        ax.text(0.12, y, line, fontsize=10, color=DARK)
        y -= 0.03

    y -= 0.02
    ax.text(0.1, y, "Key Insights", fontsize=14, fontweight="bold", color=BLUE)
    y -= 0.04
    insights = [
        "Deskewing is the single most impactful preprocessing step (45% error reduction).",
        "PCA not only speeds up inference but also improves accuracy slightly via denoising.",
        "KNN with proper preprocessing is competitive with simple neural networks on MNIST.",
        "Distance-weighted voting outperforms uniform voting consistently.",
    ]
    for line in insights:
        ax.text(0.12, y, line, fontsize=10, color=DARK)
        y -= 0.03

    y -= 0.02
    ax.text(0.1, y, "Future Work", fontsize=14, fontweight="bold", color=BLUE)
    y -= 0.04
    future = [
        "Tangent distance metric: Could achieve ~98.9% accuracy by accounting",
        "    for small transformations (rotation, scaling, thickness) in distance computation.",
        "",
        "Shape context features: Could push accuracy to ~99.4% by using",
        "    richer structural descriptors instead of raw pixel values.",
        "",
        "Ensemble methods: Combining KNN with other classifiers for even better results.",
    ]
    for line in future:
        ax.text(0.12, y, line, fontsize=10, color=DARK)
        y -= 0.028

    add_footer(fig, 13)
    return fig


def page_references():
    fig = new_page()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.92, "References", ha="center", va="center",
            fontsize=24, fontweight="bold", color=DARK)
    ax.axhline(y=0.895, xmin=0.25, xmax=0.75, color=ACCENT, linewidth=2)

    refs = [
        ("[1]  MNIST Database",
         "Yann LeCun, Corinna Cortes, Christopher J.C. Burges.",
         "http://yann.lecun.com/exdb/mnist/"),
        ("[2]  Shape Matching and Object Recognition Using Shape Contexts",
         "Serge Belongie, Jitendra Malik, Jan Puzicha (2002).",
         "IEEE Transactions on Pattern Analysis and Machine Intelligence."),
        ("[3]  Modified Sliding-Window Metric for KNN",
         "arXiv:1809.06846.",
         "https://arxiv.org/abs/1809.06846"),
        ("[4]  scikit-learn: Machine Learning in Python",
         "Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.",
         "https://scikit-learn.org/"),
        ("[5]  FastAPI Documentation",
         "Sebastian Ramirez.",
         "https://fastapi.tiangolo.com/"),
    ]

    y = 0.82
    for ref in refs:
        ax.text(0.1, y, ref[0], fontsize=12, fontweight="bold", color=DARK)
        y -= 0.03
        ax.text(0.14, y, ref[1], fontsize=10, color="gray")
        y -= 0.025
        ax.text(0.14, y, ref[2], fontsize=9, color=BLUE)
        y -= 0.045

    add_footer(fig, 14)
    return fig


def main():
    print("Generating Digit Recognition Project Report...")
    pages = [
        ("Title Page", page_title_page),
        ("Project Overview", page_overview),
        ("Preprocessing Pipeline", page_preprocessing),
        ("Sample Digits", page_sample_digits),
        ("Deskewing Comparison", page_deskew_comparison),
        ("Hyperparameter Optimization", page_hyperparameter),
        ("Model Configuration", page_model_config),
        ("Results", page_results),
        ("Per-Class Performance", page_per_class),
        ("Misclassified Examples", page_misclassified),
        ("System Architecture", page_architecture),
        ("Deployment", page_deployment),
        ("Conclusion", page_conclusion),
        ("References", page_references),
    ]

    with PdfPages(OUTPUT) as pdf:
        for name, func in pages:
            print(f"  Generating page: {name}")
            fig = func()
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

    print(f"\nReport saved to: {OUTPUT}")
    print(f"Total pages: {len(pages)}")


if __name__ == "__main__":
    main()
