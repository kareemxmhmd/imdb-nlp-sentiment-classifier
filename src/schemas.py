from pydantic import BaseModel, Field
from typing import List, Optional

class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The review text to classify")

class BatchReviewRequest(BaseModel):
    reviews: List[str] = Field(..., min_items=1, description="List of reviews to classify")

class PredictionResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    decision_tier: str
    reason: str

class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]

class TrendData(BaseModel):
    timestamp: str
    product_title: str
    total_reviews: int
    positive_rate: float
    negative_rate: float
    alert_triggered: bool

class ModelInfo(BaseModel):
    version: str
    training_date: str
    accuracy: float
    f1_score: float
    auto_tag_threshold: float
