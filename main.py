import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load model and vectorizer
with open("ensemble_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Text cleaning function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)       # Remove URLs
    text = re.sub(r'\@\w+|\#', '', text)                      # Remove mentions/hashtags
    text = re.sub(r'[^a-z\s]', '', text)                      # Remove special chars/numbers
    text = re.sub(r'\s+', ' ', text).strip()                 # Remove extra whitespace
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

# Streamlit page config
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")
st.title("Generative AI Sentiment Analysis Dashboard")

# Sidebar navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to:", ["Home", "Upload CSV", "New Text Prediction"])

# ---------------- Home Page ----------------
if option == "Home":
    st.subheader("Dashboard Overview")
    st.write("""
    This dashboard allows you to analyze social media posts related to Generative AI.
    You can visualize sentiment distribution, word clouds, and predict new text sentiment.
    """)

# ---------------- Upload CSV ----------------
elif option == "Upload CSV":
    st.subheader("Upload CSV for Analysis")
    uploaded_file = st.file_uploader("Upload a CSV file with a column named 'text'", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        if 'text' in df.columns:
            # Remove empty text rows
            df = df.dropna(subset=['text'])
            df['clean_text'] = df['text'].apply(clean_text)
            
            st.write("First 5 rows of dataset:")
            st.dataframe(df.head())
            
            # Predict sentiments
            vect_texts = vectorizer.transform(df['clean_text'])
            df['sentiment'] = model.predict(vect_texts)
            
            # Metrics
            st.subheader("Sentiment Distribution")
            sentiment_counts = df['sentiment'].value_counts()
            st.bar_chart(sentiment_counts)
            
            # Word clouds
            st.subheader("Word Clouds")
            for sentiment in ['positive', 'negative', 'neutral']:
                st.write(f"Word Cloud for {sentiment.capitalize()} Tweets")
                text_data = " ".join(df[df['sentiment'] == sentiment]['clean_text'])
                if text_data.strip() != "":
                    wc = WordCloud(width=800, height=400, background_color="white").generate(text_data)
                    plt.figure(figsize=(10,4))
                    plt.imshow(wc, interpolation="bilinear")
                    plt.axis("off")
                    st.pyplot(plt)
                else:
                    st.write("No tweets for this sentiment")
            
            # Download results
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions CSV", csv, "predictions.csv", "text/csv")
        else:
            st.error("CSV must have a column named 'text'.")

# ---------------- New Text Prediction ----------------
elif option == "New Text Prediction":
    st.subheader("Predict Sentiment of New Text")
    user_input = st.text_area("Enter text here:")
    if st.button("Predict"):
        if user_input:
            clean_input = clean_text(user_input)
            vect_input = vectorizer.transform([clean_input])
            prediction = model.predict(vect_input)[0]
            st.success(f"Sentiment: {prediction.capitalize()}")
        else:
            st.error("Please enter some text.")