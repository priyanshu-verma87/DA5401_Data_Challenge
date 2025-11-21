# DA5401 EndSem Data Challenge

Name: Priyanshu Verma  
Roll No: CH22B087  
Date: 21-11-2025  

---

## Project Overview

This project focuses on predicting the fitness score (1–10) between an AI Evaluation Metric Definition and a Prompt-Response text pair. The objective is to model semantic similarity using embeddings, feature engineering, and regression methods.

The main challenges included:
- Strong score imbalance (majority scores 8–10)
- High-dimensional metric embeddings
- Need for semantic alignment between metric definitions and prompt-response pairs

The repository includes the notebook and report describing the full workflow: preprocessing, augmentation, embeddings, model training, and test prediction.

---

## Files in this Repository

- `DA5401_EndSem_DataChallenge_clean.ipynb`: main notebook.
- `README.md`: this file.

---

## Notebook Structure

### 1. Problem Understanding
The task involves predicting a similarity score using embeddings derived from metric names and combined prompt–response text.

### 2. Exploratory Data Analysis
- Dataset includes: metric names, metric embeddings, train data, and test data.
- The `system_prompt` column contained many null values and was dropped.
- Score distribution was highly imbalanced, dominated by values between 8–10.

### 3. Data Augmentation Strategy

Two functions were used to increase low-score examples:

#### `corrupt_text(text)`
Introduces noise by:
- Truncating text  
- Dropping random words  
- Swapping sentences  

#### `make_corrupt(df, n)`
- Samples n rows  
- Keeps the prompt  
- Replaces response with corrupted text  
- Assigns low score  
- Returns synthetic low-quality dataset  

This helped balance the score distribution.

---

## Text Vectorization

### Embedding Model: `google/embeddinggemma-300m`
Used due to its strong semantic representation capability.  
Combined prompt-response text was encoded into 768-dimensional embeddings.

---

## Feature Engineering

### PCA on Metric Embeddings
Metric embeddings were reduced from 768 dimensions to 115 using PCA with 99% variance retention.

### Custom Similarity Features
Additional features were added:
- Cosine similarity
- L2 norm
- Dot product

These enhanced the relationship between text embeddings and metric embeddings.

---

## Model Training

All features (text embeddings, PCA metric embeddings, similarity scores) were stacked into a unified feature vector.

Two models were trained:

| Model      | Validation RMSE  |
|------------|------------------|
| XGBoost    | 3.45             |
| LightGBM   | 3.33             |

LightGBM performed best and was chosen for final predictions.

---

## Predictions on Test Data

The test pipeline included:
- Combine prompt + response
- Encode using embeddinggemma-300m
- Apply PCA on metric embeddings
- Compute similarity scores
- Stack all features
- Predict using LightGBM

Predicted scores were centered around 4.8 and showed a Gaussian-like distribution.

---

## Final Results

- Final RMSE on unseen test data: 3.588
- Performance reflects significant learning but leaves room for improvement.


### Required Packages

