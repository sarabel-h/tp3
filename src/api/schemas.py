from pydantic import BaseModel, Field
from typing import List, Dict

class PredictBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1)  # batch non vide

class OnePrediction(BaseModel):
    text: str
    label: int
    probabilities: Dict[str, float]

class PredictBatchResponse(BaseModel):
    results: List[OnePrediction]
    stats: Dict[str, float]  # % par classe
