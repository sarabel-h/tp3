from fastapi import FastAPI, HTTPException  # FastAPI + gestion HTTP errors
from fastapi.middleware.cors import CORSMiddleware  # middleware pour CORS
from pydantic import BaseModel, Field  # validation des données entrantes/sortantes
from typing import List, Dict  # types pour annotations
import joblib  # pour charger le modèle sauvegardé
from pathlib import Path  # pour gérer les chemins proprement

# ------- Config -------
MODEL_PATH = Path("models/sentiment_model.joblib")  # chemin vers le modèle dans le repo HF

# ------- Schémas Pydantic -------
class PredictBatchRequest(BaseModel):  # schéma de la requête POST
    texts: List[str] = Field(..., min_items=1)  # liste non vide de commentaires

class OnePrediction(BaseModel):  # schéma d’une prédiction individuelle
    text: str  # texte original
    label: int  # label prédit
    probabilities: Dict[str, float]  # proba par classe

class PredictBatchResponse(BaseModel):  # schéma de la réponse globale
    results: List[OnePrediction]  # liste des prédictions
    stats: Dict[str, float]  # stats globales (% par classe)

# ------- Chargement modèle (UNE SEULE FOIS au démarrage) -------
def load_model():  # fonction de chargement
    if not MODEL_PATH.exists():  # vérifier que le fichier existe
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")  # sinon lever erreur claire
    return joblib.load(MODEL_PATH)  # charger le modèle depuis joblib

model = load_model()  # modèle chargé une fois pour toute

# ------- App FastAPI -------
app = FastAPI(title="YouTube Sentiment API", version="1.0")  # création app

# CORS large pour extension Chrome/Edge
app.add_middleware(
    CORSMiddleware,  # middleware CORS
    allow_origins=["*"],  # autoriser toutes les origines (ok pour TP)
    allow_credentials=True,  # autoriser cookies/credentials
    allow_methods=["*"],  # autoriser toutes méthodes HTTP
    allow_headers=["*"],  # autoriser tous headers
)

@app.get("/health")  # endpoint de monitoring
def health():
    try:
        _ = model  # simple check que le modèle est chargé
        return {"status": "ok", "model_loaded": True}  # réponse OK
    except Exception as e:
        return {"status": "error", "model_loaded": False, "detail": str(e)}  # réponse erreur

@app.post("/predict_batch", response_model=PredictBatchResponse)  # endpoint batch
def predict_batch(req: PredictBatchRequest):
    try:
        texts = req.texts  # récupérer les textes envoyés

        preds = model.predict(texts)  # prédire les classes
        probas = model.predict_proba(texts)  # prédire les probabilités

        results = []  # liste des résultats
        for text, label, p in zip(texts, preds, probas):  # boucle sur chaque commentaire
            results.append({  # ajouter un dict résultat
                "text": text,  # texte
                "label": int(label),  # label en int
                "probabilities": {  # mapping classes -> probas
                    "-1": float(p[0]),
                    "0": float(p[1]),
                    "1": float(p[2]),
                }
            })

        total = len(preds)  # nombre total de commentaires
        stats = {  # proportions globales
            "-1": float((preds == -1).sum() / total),
            "0": float((preds == 0).sum() / total),
            "1": float((preds == 1).sum() / total),
        }

        return {"results": results, "stats": stats}  # renvoyer réponse finale

    except HTTPException as he:
        raise he  # remonter les erreurs FastAPI propres
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  # sinon 500 avec message clair
