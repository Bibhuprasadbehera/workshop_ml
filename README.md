# Machine Learning Workshop: Clustering & Classification

A focused hands-on machine learning repository covering unsupervised clustering techniques (K-Means, Hierarchical Clustering) and supervised classification with XGBoost, along with an interactive in-browser 2D clustering visualizer.

---

## 📂 Repository Contents

This repository contains the following core files:

```text
workshop_ml/
├── Mall_Customers.csv                      # Customer segmentation dataset
├── k_means_clustering.ipynb                # K-Means clustering with the Elbow Method
├── hierarchical_clustering.ipynb           # Agglomerative Hierarchical clustering & Dendrogram
├── XGBoost_Breast_Cancer_Classifier.ipynb  # Supervised breast cancer tumor classification
└── presentation/
    └── interactive_playground.html         # Interactive in-browser 2D clustering visualizer
```

---

## 🔍 File Details & Workflows

### 1. [`Mall_Customers.csv`](Mall_Customers.csv)
A benchmark retail customer dataset containing 200 records used for unsupervised customer segmentation:
- **Features:** `CustomerID`, `Genre` (Gender), `Age`, `Annual Income (k$)`, and `Spending Score (1-100)`.
- **Target Features for Clustering:** Typically columns `Annual Income (k$)` and `Spending Score (1-100)` are extracted to demonstrate 2D clustering intuition.

---

### 2. [`k_means_clustering.ipynb`](k_means_clustering.ipynb)
A step-by-step notebook demonstrating partitioning clustering using **K-Means**:
- **Data Preprocessing:** Extracts 2D feature matrices from `Mall_Customers.csv`.
- **Finding Optimal $k$:** Computes and plots the **Elbow Method** using Within-Cluster Sum of Squares (WCSS / Inertia) across $k = 1$ to $10$.
- **Model Training:** Initializes and fits `KMeans(n_clusters=5, init='k-means++', random_state=42)`.
- **Visualization:** Generates scatter plots displaying the 5 distinct customer personas alongside their learned cluster centroids.

---

### 3. [`hierarchical_clustering.ipynb`](hierarchical_clustering.ipynb)
Demonstrates bottom-up **Agglomerative Hierarchical Clustering**:
- **Dendrogram Analysis:** Uses `scipy.cluster.hierarchy` with Ward’s minimum variance linkage (`linkage='ward'`) to build a dendrogram and determine the optimal number of clusters.
- **Model Training:** Fits `AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')`.
- **Visualization:** Visualizes the resulting clusters and compares boundaries against K-Means.

---

### 4. [`XGBoost_Breast_Cancer_Classifier.ipynb`](XGBoost_Breast_Cancer_Classifier.ipynb)
Provides a complementary supervised learning module to contrast against unsupervised clustering:
- **Task:** Diagnostic classification of breast cancer biopsy samples into **Benign** vs. **Malignant**.
- **Model:** High-performance gradient boosting with **XGBoost** (`xgboost.XGBClassifier`).
- **Evaluation:** Evaluates training and test performance using confusion matrices, classification reports, and ROC-AUC curves.

---

### 5. [`presentation/interactive_playground.html`](presentation/interactive_playground.html)
A standalone, zero-install web application for real-time visual learning:
- **Interactive Canvas:** Draw points manually or generate synthetic point distributions (blobs, concentric rings, noisy density clusters).
- **Live Algorithms:** Run **K-Means** step-by-step (visualizing centroid updates) or **DBSCAN** directly in the browser.
- **Usage:** Open [`presentation/interactive_playground.html`](presentation/interactive_playground.html) in any modern web browser.

---

## 🚀 Getting Started

### 1. Prerequisites & Dependencies

To run the Python notebooks, install the required libraries:

```bash
pip install numpy pandas matplotlib scipy scikit-learn xgboost jupyter
```

### 2. Running the Notebooks

Launch Jupyter Notebook or JupyterLab:
```bash
jupyter notebook
```
From the browser interface, open:
- `k_means_clustering.ipynb`
- `hierarchical_clustering.ipynb`
- `XGBoost_Breast_Cancer_Classifier.ipynb`

### 3. Launching the Playground

Double-click or open [`presentation/interactive_playground.html`](presentation/interactive_playground.html) in Chrome, Firefox, Safari, or Edge.
