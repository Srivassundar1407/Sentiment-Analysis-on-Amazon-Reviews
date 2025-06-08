Amazon Product Review Sentiment Analysis

This project implements a machine learning-based sentiment analysis pipeline using Amazon product reviews. It classifies customer sentiments (Positive, Neutral, Negative) based on review text and rating data using NLP, Logistic Regression, and Random Forest models.

Project Overview:
Goal: Analyze and classify Amazon product reviews by sentiment.

Dataset: Amazon mobile phone reviews CSV dataset.

Approach: Preprocess text → Transform with TF-IDF → Train & evaluate models → Visualize insights.

Technologies Used:
1. Python, Pandas, NumPy, Matplotlib, Seaborn
2. NLTK, scikit-learn, WordCloud
3. Machine Learning models: Logistic Regression, Random Forest
4. Evaluation: Accuracy, Classification Report, ROC-AUC
5. Visualization: WordCloud, Sentiment Distribution Plot

📁 Project Structure
bash
├── amazon_sentiment_analysis.py       # Main Python script
├── amazon_reviews.csv                 # Dataset (not included in repo for size/privacy)
└── README.md                          # This file

Features:

1. Text Preprocessing: Tokenization, stopword removal, punctuation/digit cleaning
2. Sentiment Mapping: Rating → Sentiment (1-2: Negative, 3: Neutral, 4-5: Positive)
3. WordCloud: For visualizing common terms in positive reviews
4. Modeling:
    Logistic Regression with TF-IDF
    Random Forest Classifier
5. Evaluation: Accuracy, precision, recall, F1-score, ROC curves
6. Cross-Validation: K-Fold (5 splits)
7. Data Augmentation: Synonym replacement using WordNet
8. Error Analysis: Output of misclassified reviews

Sample Output:

1. Sentiment distribution bar plot
2. Word cloud of positive reviews
3. Classification reports for Logistic Regression and Random Forest
4. ROC curve comparison
5. Printed misclassified examples

References: 

Research Paper: Product Sentiment Analysis for Amazon Reviews – IJCSIT, Vol 13, No 3, June 2021

