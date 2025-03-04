# Comparative Analysis of Machine Learning Models for Breast Cancer Diagnosis Prediction

## Overview
This project focuses on the classification of breast tumors as **benign** or **malignant** using machine learning techniques. The Wisconsin Breast Cancer Dataset (WBCD) is used as a benchmark to evaluate the effectiveness of different models in medical diagnostics.

## Dataset
The **Wisconsin Breast Cancer Dataset (WBCD)** is widely recognized for medical diagnostic research. It consists of numerical features derived from fine needle aspiration (FNA) of breast masses, used to predict whether a tumor is malignant or benign.

### Data Preprocessing
To ensure optimal model performance, the dataset underwent:
- **Feature Selection**: Removal of redundant columns (e.g., ID and Unnamed: 32).
- **Handling Missing Values**: Ensuring data completeness.
- **Standardization**: Scaling features to improve model convergence.
- **Encoding**: Converting categorical target labels ('M' for Malignant, 'B' for Benign) into numerical values (1 and 0).
- **Splitting**: Dividing data into training (70%) and testing (30%) sets.

## Methodology
### Machine Learning Models Used
Three machine learning models were implemented and evaluated:
- **Logistic Regression**: Serves as a baseline model, offering high interpretability.
- **K-Nearest Neighbors (KNN)**: A distance-based algorithm that classifies samples based on nearest neighbors.
- **Random Forest (RF)**: An ensemble model that enhances prediction accuracy through decision tree aggregation.

### Model Evaluation Metrics
- **Accuracy**: Measures overall correctness of predictions.
- **Precision & Recall**: Evaluate the trade-off between false positives and false negatives.
- **F1-Score**: Balances precision and recall.
- **Confusion Matrix**: Provides insights into classification performance.

## Results
The performance of the models on the test dataset is summarized below:
| Model              | Accuracy | Precision (Benign) | Precision (Malignant) | Recall (Benign) | Recall (Malignant) | F1-Score (Benign) | F1-Score (Malignant) |
|--------------------|----------|-------------------|----------------------|----------------|------------------|------------------|--------------------|
| Logistic Regression | **98.25%** | 97% | 100% | 100% | 95% | 99% | 98% |
| K-Nearest Neighbors (KNN) | 94.74% | 94% | 97% | 98% | 89% | 96% | 93% |
| Random Forest (RF) | 94.15% | 94% | 95% | 97% | 89% | 95% | 92% |

## Installation
### Prerequisites
Ensure you have the following dependencies installed:
- Python (>=3.7)
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

