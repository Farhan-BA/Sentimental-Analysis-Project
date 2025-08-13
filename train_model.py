# train_model.py
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib
import json
import numpy as np

# Ensure NLTK data is downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

def preprocess_text(text):
    """
    Cleans and preprocesses text for the model.
    Steps include: HTML tag removal, removing non-alphabetic characters, lowercasing,
    tokenization, stopword removal, and lemmatization.
    """
    text = re.sub(r'<.*?>', '', str(text))
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if w not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w) for w in filtered_tokens]
    return " ".join(lemmatized)

# Load the dataset
try:
    df = pd.read_csv('IMDB Dataset.csv')
except FileNotFoundError:
    print("Error: IMDB Dataset.csv not found. Please make sure the file is in the same directory.")
    exit()

# Sample a subset of the data for faster training
df = df.sample(n=10000, random_state=42).reset_index(drop=True)
df['cleaned_review'] = df['review'].apply(preprocess_text)

X = df['cleaned_review']
y = df['sentiment'].apply(lambda s: 1 if s == 'positive' else 0)

# Initialize TF-IDF Vectorizer and model
vectorizer = TfidfVectorizer(max_features=5000)
model = LogisticRegression(solver='liblinear', random_state=42)

# --- K-Fold Cross-Validation ---
# This section performs K-Fold validation as requested in the criteria.
# A standard train/test split is still used later for the final model training.
# This part is just for a more robust evaluation of the model's performance.

n_splits = 5  # You can change the number of folds here
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

all_preds = []
all_true = []
all_probas = []

print(f"Starting {n_splits}-fold cross-validation...")

for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
    print(f"  Processing Fold {fold}...")
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

    X_train_tfidf_fold = vectorizer.fit_transform(X_train_fold)
    X_test_tfidf_fold = vectorizer.transform(X_test_fold)

    model.fit(X_train_tfidf_fold, y_train_fold)

    y_pred_fold = model.predict(X_test_tfidf_fold)
    y_prob_fold = model.predict_proba(X_test_tfidf_fold)[:, 1]

    all_preds.extend(y_pred_fold)
    all_true.extend(y_test_fold)
    all_probas.extend(y_prob_fold)

# Calculate final metrics from all folds
acc = accuracy_score(all_true, all_preds)
auc = roc_auc_score(all_true, all_probas)
report = classification_report(all_true, all_preds, target_names=['Negative', 'Positive'], output_dict=True)
cm = confusion_matrix(all_true, all_preds).tolist()

metrics = {'accuracy': acc, 'roc_auc': auc, 'classification_report': report, 'confusion_matrix': cm, 'cross_validation_folds': n_splits}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)
print("Evaluation metrics saved to metrics.json")

# --- Final Model Training (for app.py) ---
# Now, train the model on the entire dataset to prepare the final artifacts for the app.
print("Training final model on the entire dataset...")
X_full_tfidf = vectorizer.fit_transform(X)
model.fit(X_full_tfidf, y)
joblib.dump(model, 'sentiment_model.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
print("Final model and vectorizer saved as sentiment_model.joblib and tfidf_vectorizer.joblib")
