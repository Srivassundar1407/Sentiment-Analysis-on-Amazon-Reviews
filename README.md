Amazon Product Review Sentiment Analysis

This project implements a machine learning-based sentiment analysis pipeline using Amazon product reviews. It classifies customer sentiments (Positive, Neutral, Negative) based on review text and rating data using NLP, Logistic Regression, and Random Forest models.

Project Overview:
Goal: Analyze and classify Amazon product reviews by sentiment.
Dataset: Amazon mobile phone reviews CSV dataset.
Approach: Preprocess text → Transform with TF-IDF → Train & evaluate models → Visualize insights.

Technologies Used:
Python, Pandas, NumPy, Matplotlib, Seaborn
NLTK, scikit-learn, WordCloud
Machine Learning models: Logistic Regression, Random Forest
Evaluation: Accuracy, Classification Report, ROC-AUC
Visualization: WordCloud, Sentiment Distribution Plot

📁 Project Structure
bash
├── amazon_sentiment_analysis.py       # Main Python script
├── amazon_reviews.csv                 # Dataset (not included in repo for size/privacy)
└── README.md                          # This file

Features:
Text Preprocessing: Tokenization, stopword removal, punctuation/digit cleaning

Sentiment Mapping: Rating → Sentiment (1-2: Negative, 3: Neutral, 4-5: Positive)

WordCloud: For visualizing common terms in positive reviews

Modeling:
  Logistic Regression with TF-IDF
  Random Forest Classifier

Evaluation: Accuracy, precision, recall, F1-score, ROC curves

Cross-Validation: K-Fold (5 splits)

Data Augmentation: Synonym replacement using WordNet

Error Analysis: Output of misclassified reviews

Sample Output:
Sentiment distribution bar plot

Word cloud of positive reviews

Classification reports for Logistic Regression and Random Forest

ROC curve comparison

Printed misclassified examples

References: 
Research Paper: Product Sentiment Analysis for Amazon Reviews – IJCSIT, Vol 13, No 3, June 2021

