import sys
import os

# Ensure src is in the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_data, preprocess_data, split_and_vectorize
from src.model import train_model, evaluate_model, find_confidence_threshold, save_model

def main():
    print("Starting Training Pipeline...")
    
    # 1. Load Data
    try:
        df = load_data()
    except FileNotFoundError:
        print("Error: Could not find data/IMDB Dataset.csv. Please ensure it exists.")
        return
        
    print(f"Loaded {len(df)} records.")
    
    # 2. Preprocess
    df = preprocess_data(df)
    
    # 3. Split and Vectorize
    print("Splitting and Vectorizing data...")
    X_train_vec, X_test_vec, y_train, y_test, vectorizer = split_and_vectorize(df)
    
    # 4. Train Model
    clf = train_model(X_train_vec, y_train)
    
    # 5. Evaluate
    print("Evaluating model...")
    acc, f1 = evaluate_model(clf, X_test_vec, y_test)
    print(f"Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
    
    # 6. Threshold Tuning
    print("Tuning confidence threshold...")
    threshold = find_confidence_threshold(clf, X_test_vec, y_test, target_precision=0.98)
    print(f"Calculated Auto-tag Threshold for 98% precision: {threshold:.4f}")
    
    # 7. Save Model
    metrics = {
        "accuracy": acc,
        "f1_score": f1,
        "target_precision": 0.98,
        "auto_tag_threshold": threshold
    }
    
    metadata = {
        "dataset_size": len(df),
        "model_type": "LinearSVC + CalibratedClassifierCV",
        "features": "TF-IDF (5000 max features)"
    }
    
    version = save_model(clf, vectorizer, metrics, metadata)
    print(f"Pipeline completed successfully. Model saved as version {version}.")

if __name__ == "__main__":
    main()
