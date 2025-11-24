import joblib
from pathlib import Path
from fastapi import HTTPException

MODEL_PATH = Path("models/sentiment_model.joblib")

class SentimentModel:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")
        self.model = joblib.load(MODEL_PATH)  # charge une seule fois au démarrage

    def predict_many(self, texts):
        if not isinstance(texts, list) or len(texts) == 0:
            raise HTTPException(status_code=400, detail="Batch vide ou invalide.")

        preds = self.model.predict(texts)  # labels
        probas = self.model.predict_proba(texts)  # proba par classe

        results = []
        for text, label, p in zip(texts, preds, probas):
            results.append({
                "text": text,
                "label": int(label),
                "probabilities": {
                    "-1": float(p[0]),
                    "0": float(p[1]),
                    "1": float(p[2]),
                }
            })

        # stats globales (%)
        total = len(preds)
        stats = {
            "-1": float((preds == -1).sum() / total),
            "0": float((preds == 0).sum() / total),
            "1": float((preds == 1).sum() / total),
        }

        return results, stats

sentiment_model = SentimentModel()
