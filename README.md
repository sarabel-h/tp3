# YouTube Sentiment Analyzer

Analyse automatique des commentaires YouTube via un pipeline Machine Learning, une API FastAPI et une extension Chrome.  
Projet réalisé dans le cadre du module Virtualisation & Cloud Computing (TP3).

## 1. Description du projet

Ce projet fournit une solution complète permettant :
- d'extraire les commentaires visibles sous une vidéo YouTube,
- de prédire leur sentiment (positif, neutre, négatif),
- d'afficher les résultats dans une extension Chrome,
- d'exposer un modèle entraîné via une API FastAPI,
- de déployer cette API à l’aide d’une image Docker sur Hugging Face Spaces.

Le système repose sur un pipeline reproductible, une API robuste et une architecture adaptée au déploiement cloud.

## 2. Architecture technique

### Vue d’ensemble
YouTube → Extension Chrome → API FastAPI → Modèle ML → Résultats

### Structure du projet
```
TP3/
├── data/
├── src/
│   ├── data/                 # Prétraitement, EDA
│   ├── models/               # Entraînement et sauvegarde
│   └── api/                  # Code FastAPI
├── models/                   # Modèle final
├── chrome-extension/         # Extension Chrome
├── youtube-sentiment-api/    # Déploiement Docker HuggingFace
├── requirements_dev.txt
└── README.md
```

## 3. Pipeline Machine Learning

Étapes principales :
- Nettoyage des textes (URLs, mentions, ponctuation, stopwords),
- Analyse exploratoire (distribution des classes, longueurs, vocabulaire),
- Tests de modèles (Logistic Regression, SVM, Random Forest),
- Sélection du modèle final : TF-IDF + Logistic Regression.

Performances :
- Accuracy : 88 %
- Temps d’inférence moyen : < 100 ms

## 4. API FastAPI

Endpoints :
- `GET /health` : vérifie l'état de l'API,
- `POST /predict_batch` : prédictions par lot.

Exemple d'appel :
```json
{
  "texts": ["This is a great video."]
}
```

## 5. Extension Chrome

Fonctionnalités :
- extraction automatique des commentaires YouTube,
- envoi des textes à l’API,
- affichage filtré des résultats,
- interface simple et adaptée à l’usage réel.

Structure :
```
chrome-extension/
├── manifest.json
├── popup.html
├── popup.js
├── content.js
└── styles.css
```

## 6. Déploiement Docker et Hugging Face

Image utilisée : Python 3.10-slim.  
Le répertoire `youtube-sentiment-api/` contient :
- un Dockerfile,
- l’application FastAPI,
- les dépendances nécessaires.

Déploiement local :
```bash
docker build -t youtube-sentiment-api .
docker run -p 7860:7860 youtube-sentiment-api
```

## 7. Installation et utilisation

### Installation des dépendances
```
pip install -r requirements_dev.txt
```

### Lancement de l’API en local
```
uvicorn app_api:app --host 0.0.0.0 --port 7860
```

Accès à la documentation :  
http://localhost:7860/docs

### Installation de l’extension Chrome
1. Ouvrir chrome://extensions  
2. Activer le mode développeur  
3. Charger le dossier `chrome-extension/`


