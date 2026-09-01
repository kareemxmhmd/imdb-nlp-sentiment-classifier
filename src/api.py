import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from src.schemas import ReviewRequest, BatchReviewRequest, PredictionResponse, BatchPredictionResponse, TrendData, ModelInfo
from src.data import clean_text, normalize_text
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import random

# Load nltk resources for prediction
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

app = FastAPI(title="ReviewPulse API", description="Sentiment Classification & Decision Triage", version="1.0.0")

# Globals for model artifacts
model = None
vectorizer = None
metadata = None
threshold = 0.85

def load_model_artifacts():
    global model, vectorizer, metadata, threshold
    model_dir = "models"
    try:
        model = joblib.load(os.path.join(model_dir, "model_latest.pkl"))
        vectorizer = joblib.load(os.path.join(model_dir, "vectorizer_latest.pkl"))
        with open(os.path.join(model_dir, "metadata_latest.json"), "r") as f:
            metadata = json.load(f)
        threshold = metadata.get("metrics", {}).get("auto_tag_threshold", 0.85)
        print("Model artifacts loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load model artifacts. {e}")

@app.on_event("startup")
async def startup_event():
    load_model_artifacts()

def process_and_predict(text: str) -> PredictionResponse:
    if not model or not vectorizer:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    cleaned = clean_text(text)
    normalized = normalize_text(cleaned, stop_words, lemmatizer)
    
    vec_input = vectorizer.transform([normalized])
    probs = model.predict_proba(vec_input)[0]
    
    # Classes are usually [0, 1] where 1 is positive
    # Check classes_ attribute to be sure, assuming 1 is positive
    is_positive = model.classes_[1] == 1
    pos_idx = 1 if is_positive else 0
    neg_idx = 0 if is_positive else 1
    
    pos_prob = probs[pos_idx]
    neg_prob = probs[neg_idx]
    
    confidence = max(pos_prob, neg_prob)
    sentiment = "Positive" if pos_prob > neg_prob else "Negative"
    
    if confidence >= threshold:
        decision_tier = f"Auto-{sentiment}"
        reason = f"Confidence {confidence:.2f} >= threshold {threshold:.2f}"
    else:
        decision_tier = "Needs Human Review"
        reason = f"Confidence {confidence:.2f} < threshold {threshold:.2f}"
        
    return PredictionResponse(
        text=text,
        sentiment=sentiment,
        confidence=confidence,
        decision_tier=decision_tier,
        reason=reason
    )

@app.post("/predict", response_model=PredictionResponse)
def predict_single(request: ReviewRequest):
    return process_and_predict(request.text)

@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchReviewRequest):
    results = [process_and_predict(text) for text in request.reviews]
    return BatchPredictionResponse(results=results)

@app.get("/trends")
def get_trends():
    """
    Simulated trends data for dashboard.
    """
    products = ["Movie A", "Movie B", "Movie C"]
    trends = []
    now = datetime.now()
    
    for i in range(14): # Last 14 days
        date = (now - timedelta(days=13-i)).strftime("%Y-%m-%d")
        for product in products:
            # Inject a simulated spike for Movie B on day 10
            if product == "Movie B" and i == 10:
                pos_rate = random.uniform(0.1, 0.3)
                neg_rate = 1.0 - pos_rate
                alert = True
                total = random.randint(500, 1000)
            else:
                pos_rate = random.uniform(0.5, 0.9)
                neg_rate = 1.0 - pos_rate
                alert = False
                total = random.randint(100, 500)
                
            trends.append(TrendData(
                timestamp=date,
                product_title=product,
                total_reviews=total,
                positive_rate=pos_rate,
                negative_rate=neg_rate,
                alert_triggered=alert
            ))
            
    return trends

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/model/info", response_model=ModelInfo)
def model_info():
    if not metadata:
        raise HTTPException(status_code=404, detail="Model metadata not found")
        
    metrics = metadata.get("metrics", {})
    return ModelInfo(
        version=metadata.get("version", "unknown"),
        training_date=metadata.get("training_date", "unknown"),
        accuracy=metrics.get("accuracy", 0.0),
        f1_score=metrics.get("f1_score", 0.0),
        auto_tag_threshold=metrics.get("auto_tag_threshold", 0.0)
    )
