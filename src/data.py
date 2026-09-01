import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def load_data(filepath="data/IMDB Dataset.csv"):
    df = pd.read_csv(filepath)
    return df

def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<[^>]*>', ' ', text)
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Lowercase
    text = text.lower()
    return text

def normalize_text(text, stop_words, lemmatizer):
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

def preprocess_data(df):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    print("Cleaning text...")
    df['cleaned_review'] = df['review'].apply(clean_text)
    
    print("Normalizing text (this might take a while)...")
    df['normalized_review'] = df['cleaned_review'].apply(lambda x: normalize_text(x, stop_words, lemmatizer))
    
    # Convert sentiment to binary
    df['label'] = (df['sentiment'] == 'positive').astype(int)
    
    return df

def split_and_vectorize(df, test_size=0.2, random_state=42):
    X = df['normalized_review']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    return X_train_vec, X_test_vec, y_train, y_test, vectorizer
