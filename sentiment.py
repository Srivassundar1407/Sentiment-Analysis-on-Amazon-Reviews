import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab/english')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')

data = pd.read_csv(r"C:\Users\sriva\Downloads\aiml_project\amazon_reviews.csv")


stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  
    text = re.sub(r'[^\w\s]', '', text)  
    text = re.sub(r'\s+', ' ', text).strip()  
    words = word_tokenize(text) 
    words = [word for word in words if word not in stop_words]  
    return ' '.join(words)


data['Cleaned_Review'] = data['reviews.text'].apply(preprocess_text)

def assign_sentiment(rating):
    if rating in [4, 5]:
        return 'Positive'
    elif rating in [1, 2]:
        return 'Negative'
    else:
        return 'Neutral'

data['Sentiment'] = data['reviews.rating'].apply(assign_sentiment)


filtered_data = data[data['Sentiment'].isin(['Positive', 'Neutral', 'Negative'])]
sentiment_mapping = {'Positive': 2, 'Neutral': 1, 'Negative': 0}
filtered_data['Sentiment'] = filtered_data['Sentiment'].map(sentiment_mapping)

plt.figure(figsize=(8, 5))
sns.countplot(x=filtered_data['Sentiment'].map({0: 'Negative', 1: 'Neutral', 2: 'Positive'}), palette='viridis')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

positive_text = ' '.join(filtered_data[filtered_data['Sentiment'] == 2]['Cleaned_Review'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Positive Sentiment Word Cloud')
plt.show()

X = filtered_data['Cleaned_Review']
y = filtered_data['Sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('classifier', LogisticRegression(max_iter=1000, class_weight=class_weight_dict))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X, y, cv=kf, scoring='accuracy')
print(f"K-Fold Cross-Validation Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

rf_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
])

rf_pipeline.fit(X_train, y_train)
rf_y_pred = rf_pipeline.predict(X_test)
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_y_pred))
print("Accuracy:", accuracy_score(y_test, rf_y_pred))

def plot_multiclass_roc(model, X_test, y_test, class_names):
    y_prob = model.predict_proba(X_test)
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_test == i, y_prob[:, i])
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc_score(y_test == i, y_prob[:, i]):.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

class_names = ['Negative', 'Neutral', 'Positive']
plot_multiclass_roc(pipeline, X_test, y_test, class_names)

from nltk.corpus import wordnet

def synonym_replacement(text, n=2):
    words = text.split()
    for _ in range(n):
        word_idx = np.random.randint(0, len(words))
        synonyms = wordnet.synsets(words[word_idx])
        if synonyms:
            words[word_idx] = synonyms[0].lemmas()[0].name()
    return ' '.join(words)

augmented_text = synonym_replacement(X_train.iloc[0])
print("Original Text:", X_train.iloc[0])
print("Augmented Text:", augmented_text)

errors = X_test[y_test != y_pred]
error_analysis = pd.DataFrame({'Review': errors, 'Actual': y_test[y_test != y_pred], 'Predicted': y_pred[y_test != y_pred]})
print("Sample Misclassified Reviews:")
print(error_analysis.head())
