import os
import joblib
import json
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, classification_report

def train_model(X_train, y_train):
    print("Training base LinearSVC model...")
    base_clf = LinearSVC(random_state=42, dual=False) # dual=False when n_samples > n_features, but with TF-IDF n_features is large. Actually, better use default or dual='auto' for sklearn >= 1.3
    
    # Let's just use SVC(probability=True) or CalibratedClassifierCV
    print("Calibrating model...")
    clf = CalibratedClassifierCV(estimator=base_clf, method='sigmoid', cv=3)
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(clf, X_test, y_test):
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    return acc, f1

def find_confidence_threshold(clf, X_val, y_val, target_precision=0.98):
    """
    Finds a threshold for the confidence score (max probability) such that 
    the precision on the auto-tagged predictions meets the target.
    """
    probs = clf.predict_proba(X_val)
    confidences = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    
    # We want to find a threshold T such that if confidence >= T, precision is >= target_precision
    # We will test thresholds from 0.5 to 0.99
    thresholds = np.linspace(0.5, 0.99, 50)
    best_threshold = 0.99
    
    for t in thresholds:
        auto_mask = confidences >= t
        if np.sum(auto_mask) == 0:
            continue
        
        auto_preds = preds[auto_mask]
        auto_labels = y_val.to_numpy()[auto_mask] if isinstance(y_val, pd.Series) else y_val[auto_mask]
        
        # Calculate precision for both classes in auto tier
        # Weighted precision might be good, or just overall accuracy on this subset
        auto_acc = accuracy_score(auto_labels, auto_preds)
        
        if auto_acc >= target_precision:
            best_threshold = t
            break
            
    return best_threshold

def save_model(clf, vectorizer, metrics, metadata, model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_path = os.path.join(model_dir, f"model_{version}.pkl")
    vec_path = os.path.join(model_dir, f"vectorizer_{version}.pkl")
    meta_path = os.path.join(model_dir, f"metadata_{version}.json")
    
    print(f"Saving model artifacts to {model_dir}...")
    joblib.dump(clf, model_path)
    joblib.dump(vectorizer, vec_path)
    
    meta_info = {
        "version": version,
        "training_date": datetime.now().isoformat(),
        "metrics": metrics,
        "metadata": metadata
    }
    
    with open(meta_path, "w") as f:
        json.dump(meta_info, f, indent=4)
        
    # Also save a 'latest' pointer or copy
    joblib.dump(clf, os.path.join(model_dir, "model_latest.pkl"))
    joblib.dump(vectorizer, os.path.join(model_dir, "vectorizer_latest.pkl"))
    with open(os.path.join(model_dir, "metadata_latest.json"), "w") as f:
        json.dump(meta_info, f, indent=4)
        
    return version
