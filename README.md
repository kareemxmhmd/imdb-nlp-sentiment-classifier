# IMDB NLP Sentiment Classifier

## Overview
Built a sentiment analysis model for IMDB movie reviews, automatically classifying reviews as **positive** or **negative**. The project benchmarks 10 ML algorithms across 3 feature engineering strategies on 50,000 reviews.

## Dataset

The dataset used is the [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) from Kaggle.

- 50,000 movie reviews labeled as positive or negative
- Download the CSV and place it in the root directory as `IMDB Dataset.csv`

## Technical Approach
A comprehensive NLP pipeline was implemented:

1. **Data Preprocessing**: 
   - Text cleaning (HTML tags, special characters)
   - Stopword removal and lemmatization
   - Train/test split (50,000 reviews total)

2. **Feature Engineering**:
   - Bag-of-Words (BoW) vectorization
   - TF-IDF vectorization  
   - Word2Vec embeddings

3. **Model Comparison**: Tested 10 ML algorithms:
   - Logistic Regression, Naive Bayes, Linear SVC
   - Random Forest, Gradient Boosting , SGD Classifier
   - Decision Tree, KNN, Passive Aggressive

## Results
- **Best Performance**: Linear SVC + TF-IDF achieved **90.09% accuracy**
- **Fastest**: Bernoulli Naive Bayes at 87.35% accuracy (0.1s)
- **Alternative**: Word2Vec + Logistic Regression reached 85.45%

## Key Takeaways
- TF-IDF consistently outperformed BoW and Word2Vec for classical ML models
- Linear SVC is the best balance of accuracy and speed for this task
- Word2Vec embeddings shine with larger datasets and deep learning models