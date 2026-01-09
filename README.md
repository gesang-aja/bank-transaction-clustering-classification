# Bank Transaction Clustering and Classification

## 📌 Project Overview

This project is the **final submission for the Machine Learning course**, designed to demonstrate the integration of **unsupervised learning (clustering)** and **supervised learning (classification)** in a single, end-to-end machine learning workflow.

The dataset used in this project is a modified version of the *Bank Transaction Dataset for Fraud Detection*, provided by the course to meet specific evaluation criteria. The main objective is to:

1. Generate labels from **unlabeled transaction data** using clustering.
2. Use the generated labels as targets for **classification models**.

This project follows the official submission template and evaluation rubric provided in the course.

---

## 🎯 Project Objectives

* Perform **Exploratory Data Analysis (EDA)** on bank transaction data
* Apply **data preprocessing and feature engineering**
* Generate transaction labels using **K-Means clustering**
* Interpret and analyze clustering results
* Build **classification models** to predict cluster labels
* Evaluate and compare supervised learning models

---

## 🗂 Dataset Information

* **Source**: Course-provided Google Drive dataset
* **Original Reference**: Kaggle – Bank Transaction Dataset for Fraud Detection
* **Condition**: Dataset used in this project is **modified** and must match the course-provided version

### Dropped Columns

The following columns are removed during preprocessing as they represent identifiers or timestamps:

* TransactionID
* AccountID
* DeviceID
* IPAddress
* MerchantID
* TransactionDate

---

## 🔍 Exploratory Data Analysis (EDA)

The EDA process includes:

* Displaying dataset samples using `head()`
* Inspecting data types and structure with `info()`
* Generating statistical summaries using `describe()`
* Correlation matrix visualization
* Histogram visualizations for numerical and categorical features

Special care is taken to ensure visualizations are **clear, readable, and non-overlapping**, following best practices.

---

## 🧹 Data Preprocessing

The preprocessing pipeline consists of:

* Handling missing values using `dropna()`
* Removing duplicate records using `drop_duplicates()`
* Encoding categorical features using `LabelEncoder`
* Outlier handling using row removal (drop method)
* Feature scaling using `StandardScaler`
* Feature binning on selected numerical features (Advanced step)

---

## 🔗 Clustering Stage (Unsupervised Learning)

### Methods Used

* **Algorithm**: K-Means Clustering
* **Cluster Selection**: Elbow Method using `KElbowVisualizer`
* **Dimensionality Reduction (Advanced)**: Principal Component Analysis (PCA)

### Evaluation Metrics

* Inertia (Elbow Method)
* Silhouette Score

### Outputs

* Trained clustering model saved as:

  ```
  model_clustering
  ```
* PCA comparison model saved as:

  ```
  PCA_model_clustering.h5
  ```

---

## 📊 Clustering Interpretation

* Visual analysis of clustering results
* Descriptive statistics (mean, min, max) for numerical features
* Mode analysis for categorical features (after inverse transform)
* Interpretation of cluster characteristics

### Generated Target

* Cluster labels are stored in a new column named:

  ```
  Target
  ```

### Exported Dataset

* Inverse-transformed dataset with cluster labels:

  ```
  data_clustering_inverse.csv
  ```

---

## 🤖 Classification Stage (Supervised Learning)

### Data Splitting

* Dataset split using `train_test_split()`

### Models Implemented

* **Decision Tree Classifier** (baseline model)
* Additional classification models

### Model Evaluation

Each model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score

### Saved Models

* Decision Tree model:

  ```
  decision_tree_model.h5
  ```
* Additional explored models:

  ```
  explore_knn_classification
  ```
* Tuned classification model:

  ```
  tuning_classification
  ```

---

## 🛠 Tools & Libraries

* Python
* NumPy, Pandas
* Matplotlib, Seaborn
* Scikit-learn (recommended version: **1.7.0**)
* Joblib
* Yellowbrick

---

## 📁 Project Structure

```
bank-transaction-clustering-classification/
│
├── notebook.ipynb
├── data/
│   └── dataset_clustering_project.csv
├── models/
│   ├── model_clustering
│   ├── PCA_model_clustering.h5
│   ├── decision_tree_model.h5
│   └── tuning_classification
├── data_clustering_inverse.csv
└── README.md
```
---

## 🏁 Conclusion

This project demonstrates a complete machine learning workflow by integrating clustering and classification techniques. It highlights the importance of unsupervised learning for label generation and supervised learning for predictive modeling in real-world, unlabeled datasets.

---

📌 *This repository is intended for educational and evaluation purposes as part of a machine learning course submission.*

