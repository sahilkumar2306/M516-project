import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
df = pd.read_csv('Tweets.csv')

# Text preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

df['clean_text'] = df['text'].apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['sentiment'], test_size=0.2, random_state=42, stratify=df['sentiment']
)

# TF-IDF vectorizer
tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1,3))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Ensemble model
rf_clf = RandomForestClassifier(n_estimators=500, max_depth=50, random_state=42, class_weight='balanced')
xgb_clf = xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
lr_clf = LogisticRegression(max_iter=500, class_weight='balanced')

ensemble = VotingClassifier(
    estimators=[('rf', rf_clf), ('xgb', xgb_clf), ('lr', lr_clf)],
    voting='soft'
)

# Train model
ensemble.fit(X_train_tfidf, y_train)

# Save model and vectorizer
with open("ensemble_model.pkl", "wb") as f:
    pickle.dump(ensemble, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("Model and vectorizer saved successfully!")