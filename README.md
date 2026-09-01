# ReviewPulse: Sentiment Classification & Decision Triage

## Overview
ReviewPulse is an NLP-powered sentiment classification system built for Platform Trust & Safety and Customer Insights teams. Rather than just blindly labeling text, this system introduces a **Confidence-Based Decision Triage** layer. It automatically tags highly confident predictions and routes ambiguous reviews to a human-in-the-loop queue, ensuring both scale and accuracy.

This project was built using the [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and transitions a benchmarking ML notebook into a full-stack, production-ready application.

## Features
- **Calibrated Sentiment Classification:** Utilizes a `LinearSVC` model with TF-IDF features and Platt scaling (`CalibratedClassifierCV`) to output precise confidence probabilities.
- **Decision Triage Logic:** 
  - Automatically tags predictions with >= 98% precision (e.g., Confidence > 0.90).
  - Routes lower-confidence predictions to a "Needs Human Review" tier.
- **REST API (FastAPI):** High-performance endpoints for single predictions, batch processing, and trend aggregations.
- **Interactive Dashboard (Streamlit):** A lightweight UI to test predictions, process CSV batch uploads, and monitor sentiment trends.
- **Production Ready:** Includes Dockerization, CI/CD GitHub Actions workflow, and a structured model registry.

## Explicit Scope Boundaries
To provide clarity on what is real versus simulated in this portfolio product:
- **Real:** The dataset, text processing pipeline, sentiment labels, model training/benchmarking, probability calibration, decision-tiering logic, API, UI, and deployment configuration.
- **Simulated:** The original dataset lacks timestamps and product/title metadata. To demonstrate the dashboard's trend monitoring and spike alert capabilities, synthetic dates and product titles are generated dynamically by the API.

## Project Structure
```text
.
├── data/               # Raw IMDB Dataset.csv (not tracked in git)
├── models/             # Serialized model artifacts and metadata
├── src/
│   ├── data.py         # Data ingestion and preprocessing pipeline
│   ├── model.py        # Model training, calibration, and threshold tuning
│   ├── train.py        # Main execution script to train and save the model
│   ├── api.py          # FastAPI application
│   ├── schemas.py      # Pydantic models for API validation
│   └── ui.py           # Streamlit dashboard
├── tests/              # Unit tests
├── Dockerfile          # Containerization for the API
└── requirements.txt    # Project dependencies
```

## Setup & Installation

### 1. Data Preparation
1. Download the [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).
2. Place the extracted `IMDB Dataset.csv` inside the `data/` directory.

### 2. Environment Setup
```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Model Training
Run the training pipeline to preprocess the data, train the model, tune the threshold, and save the artifacts:
```bash
python src/train.py
```
*(Expected performance: ~88.7% Accuracy, Auto-tag threshold tuned for 98% precision)*

### 4. Running the Application
**Start the FastAPI Backend:**
```bash
uvicorn src.api:app --reload
```
*API docs available at `http://localhost:8000/docs`*

**Start the Streamlit UI Dashboard:**
```bash
streamlit run src/ui.py
```
*Dashboard opens automatically in your browser.*

## Prior Benchmarking
Before productization, 10 ML algorithms and 3 feature representations (BoW, TF-IDF, Word2Vec) were benchmarked. 
- **Best overall:** `Linear SVC` + `TF-IDF` (selected for production).
- **Fastest:** `Bernoulli Naive Bayes`.
- *Details of this benchmarking phase can be found in the original `imdb-nlp.ipynb` notebook.*