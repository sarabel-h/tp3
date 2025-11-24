from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.api.schemas import PredictBatchRequest, PredictBatchResponse
from src.api.predict import sentiment_model

app = FastAPI(title="Reddit Sentiment API", version="1.0")

# CORS pour extension Chrome :contentReference[oaicite:2]{index=2}
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en dev: tout autoriser
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")  # endpoint monitoring :contentReference[oaicite:3]{index=3}
def health():
    try:
        _ = sentiment_model.model  # check modèle chargé
        return {"status": "ok", "model_loaded": True}
    except Exception as e:
        return {"status": "error", "model_loaded": False, "detail": str(e)}

@app.post("/predict_batch", response_model=PredictBatchResponse)  # seul endpoint prédiction :contentReference[oaicite:4]{index=4}
def predict_batch(req: PredictBatchRequest):
    try:
        results, stats = sentiment_model.predict_many(req.texts)
        return {"results": results, "stats": stats}
    except HTTPException as he:
        raise he
    except Exception as e:
        # gestion d'erreur propre :contentReference[oaicite:5]{index=5}
        raise HTTPException(status_code=500, detail=str(e))
