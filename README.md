# 🐦 Bird Species Classification in Natural Soundscapes https://www.kaggle.com/competitions/birdclef-2026

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Librosa](https://img.shields.io/badge/Librosa-Audio-orange)](https://librosa.org/)
[![Google Perch](https://img.shields.io/badge/Google%20Perch-Transfer%20Learning-green)](https://github.com/google-research/perch)


> Multi-label acoustic classification across **234 bird species** under extreme long-tail class imbalance and environmental polyphony, benchmarking expert acoustic features (**MFCC**) against pre-trained bioacoustic representations (**Google Perch**).

📄 **[Read the Full Technical Report (PDF)](./report.pdf)**

---

## 📌 Problem & Motivation

Passive Acoustic Monitoring (PAM) is crucial for automated avian biodiversity monitoring. However, environmental bioacoustics faces major technical bottlenecks:
* **Severe Background Noise & Polyphony:** Dense ambient soundscapes (wind, rain, overlapping vocalizations) with low Signal-to-Noise Ratio (SNR).
* **Extreme Long-Tail Distribution:** A few dominant species possess thousands of recordings, while rare species have as few as 1 to 3 samples, creating a few-shot learning challenge.
* **Strict Evaluation Protocol:** Chunk-level multi-label classification on 5-second windows (234 binary targets per window). To prevent microphone-specific acoustic data leakage, all splits strictly adhere to a file-grouped partitioning (`GroupShuffleSplit`).

---

## 🔬 Experimental Methodology

### 1. Feature Representations
* **Handcrafted Acoustic Features (50 MFCCs):** Extracted using `Librosa` and `SciPy` to capture spectral envelopes and timbre without relying on external pre-training biases.
* **Bioacoustic Transfer Learning (Google Perch - 1280D):** High-dimensional semantic embeddings from Google's deep convolutional bioacoustic foundation model.

### 2. Modeling Paradigms & Algorithms
* **Linear Baseline:** Logistic Regression via `OneVsRestClassifier` (234 independent binary heads).
* **Non-Linear Ensemble:** Random Forest with constrained tree depth (`max_depth=60`, `n_estimators=50`) to limit memory overhead over 200,000 samples.
* **Kernel Approximation:** Nystroëm RBF kernel projection ($50 \to 300$ dimensions) to test non-linear separability.
* **Semi-Supervised Learning:** Pseudo-labeling on unannotated soundscapes with a high confidence threshold ($p > 0.90$).
* **Cost-Sensitive Learning:** Balanced class weighting (`class_weight='balanced'`) systematically enforced across all loss functions.

---

## 📊 Benchmark Results

Performance evaluated on **Macro F1-Score** across Train, Validation, and Test sets:

| Model | Feature Space | F1-Train | F1-Val | F1-Test | Average Precision (AP) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Random Forest | 50 MFCC | 0.4813 | 0.3750 | 0.0128 | — |
| Linear LogReg (OvR) | 50 MFCC | 0.0890 | 0.0482 | 0.0377 | 0.04 |
| RBF Nystroëm (Kernel) | 50 MFCC | 0.0658 | 0.0517 | 0.0340 | — |
| **LogReg (Google Perch)** | **1280D Perch** | **0.2735** | **0.2223** | **0.2116** | **0.67** |
| LogReg (Perch + Pseudo-Label) | 1280D Perch | 0.2466 | 0.1985 | 0.1864 | — |

---

## 💡 Key Findings

* **Invalidation of Non-Linear Separability on MFCCs:** Non-linear models failed to outperform the linear baseline. Random Forest severely overfitted noise ($0.4813 \to 0.0128$), while Nystroëm RBF created tiny local spheres of influence around rare points, failing to extrapolate to testing variants.
* **Transfer Learning Breakthrough:** Google Perch embeddings resolved semantic ambiguity in ambient noise, increasing test Macro-F1 by **$6\times$** ($0.0377 \to 0.2116$) and pushing Average Precision to **0.67**.
* **Failure of Pseudo-Labeling:** High-confidence pseudo-labels clustered exclusively around majority classes (as shown via UMAP projections), reinforcing representation bias and degrading test performance.

---

## 🛠️ Repository Structure

```text
├── raw_data/                # Raw audio soundscapes (.ogg / .wav)
├── processed_data4/         # 5-second windowed segmented data
├── features/
│   ├── manual_embedding/    # 50 MFCC feature vectors
│   └── perch_embedding/     # 1280D Google Perch embeddings
├── scripts/
│   ├── embedding_perch.py   # Perch model inference & embedding extraction
│   ├── main.py              # Training, hyperparameter grid search, and evaluation
│   └── utils.py             # Data loader, metrics, and visualization scripts
├── ressource/results/       # Confusion matrices, PR curves, and UMAP projections
├── report.pdf               # Full academic synthesis report
└── README.md

h
