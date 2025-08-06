# Spam Message Classification using Machine Learning

## Overview

Spam detection is one of the most well-known applications of natural language processing (NLP) and machine learning. This project aims to build an accurate and efficient classifier that can automatically distinguish between **spam** and **ham (non-spam)** messages using classical machine learning algorithms.

The core components of this repository include data cleaning, feature engineering, model training, evaluation, and performance analysis. The entire pipeline is implemented in a modular, reproducible Jupyter Notebook format.

This project is ideal for learning about:

- Text preprocessing and feature extraction
- Supervised classification
- Dealing with imbalanced datasets
- Evaluation metrics for classification tasks

## Table of Contents

- [Spam Message Classification using Machine Learning](#spam-message-classification-using-machine-learning)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Dataset](#dataset)
  - [Problem Statement](#problem-statement)
  - [Objectives](#objectives)
  - [Installation](#installation)
  - [Project Structure](#project-structure)
  - [Detailed Methodology](#detailed-methodology)
    - [1. Data Exploration](#1-data-exploration)
    - [2. Text Preprocessing](#2-text-preprocessing)
    - [3. Feature Extraction](#3-feature-extraction)
    - [4. Model Training](#4-model-training)
    - [5. Evaluation Metrics](#5-evaluation-metrics)
  - [Model Comparison](#model-comparison)
  - [Results and Insights](#results-and-insights)
  - [Limitations](#limitations)
  - [Future Improvements](#future-improvements)

## Dataset

The dataset used in this project is the **SMS Spam Collection Dataset**, a well-known public corpus made available by the UCI Machine Learning Repository. It contains 5,574 English messages labeled as either `spam` or `ham`.

- **Attributes**:
  - `label`: Indicates whether the message is spam or ham
  - `message`: The text content of the message

## Problem Statement

Email and SMS spam pose a significant threat to digital communication. Traditional rule-based spam filters are limited in their adaptability and are often bypassed by increasingly sophisticated spam techniques. This project aims to create a **machine learning pipeline** that is:

- Adaptable
- Generalizable
- Lightweight
- Easily deployable

## Objectives

- Load and inspect the dataset
- Perform exploratory data analysis
- Clean and preprocess the text data
- Convert text to numerical format using vectorization techniques
- Train and compare multiple classifiers
- Evaluate using appropriate performance metrics
- Optimize the best model for maximum generalization
- Prepare the model for deployment and real-world use

## Installation

Clone this repository to your local machine and install the necessary dependencies.

```bash
git clone https://github.com/Adityajha3/spam-classification.git
cd spam-classification
pip install -r requirements.txt
```

## Project Structure

```
spam-classification/
│
├── spamClassification.ipynb      # Main project notebook
├── README.md                         # Project documentation
├── requirements.txt                  # List of dependencies
├── dataset/                          # Folder containing dataset files
  └── spam.csv
```

## Detailed Methodology

### 1. Data Exploration

- Class distribution
- Message length analysis
- Most frequent terms in spam vs ham

### 2. Text Preprocessing

- Lowercasing
- Removing punctuation and digits
- Stopword removal
- Tokenization
- Stemming

### 3. Feature Extraction

Text is transformed into numerical vectors using:

- CountVectorizer
- TF-IDF

### 4. Model Training

- Multinomial Naive Bayes
- Logistic Regression
- SVM
- Decision Tree
- Random Forest

### 5. Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Model Comparison

| Model               | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| Naive Bayes         | 97.8%    | 93.6%     | 95.4%  | 94.5%    |
| Logistic Regression | 98.3%    | 96.1%     | 95.7%  | 95.9%    |
| SVM                 | 98.2%    | 95.8%     | 95.3%  | 95.5%    |

## Results and Insights

- Most spam messages are longer and contain specific trigger words.
- The models performed significantly better with TF-IDF.
- Naive Bayes was fast and lightweight.
- Logistic Regression and SVM gave higher accuracy.

## Limitations

- Dataset limited to English messages.
- No contextual awareness.
- Not trained on multilingual or modern spam formats.

## Future Improvements

- Use deep learning models like LSTM or transformers.
- Deploy the model as an app or API.
- Expand the dataset to include new languages and formats.
