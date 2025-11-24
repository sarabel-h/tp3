import pandas as pd  # lire les CSV train/test
from pathlib import Path  # gérer les chemins proprement
from sklearn.feature_extraction.text import TfidfVectorizer  # transformer texte -> vecteurs TF-IDF
from sklearn.linear_model import LogisticRegression  # modèle de classification
from sklearn.pipeline import Pipeline  # enchaîner TF-IDF + modèle dans un pipeline
from sklearn.model_selection import GridSearchCV  # chercher les meilleurs hyperparamètres
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # métriques d’évaluation
import joblib  # sauvegarder le modèle et le vectoriseur

# --- Chemins des fichiers ---
TRAIN_PATH = Path("data/processed/train.csv")  # dataset d'entraînement
TEST_PATH = Path("data/processed/test.csv")  # dataset de test
MODEL_DIR = Path("models")  # dossier où stocker le modèle final
MODEL_PATH = MODEL_DIR / "sentiment_model.joblib"  # fichier modèle+vectoriseur sauvegardé

def train_model():
    # --- Charger les données ---
    train_df = pd.read_csv(TRAIN_PATH)  # lit train.csv
    test_df = pd.read_csv(TEST_PATH)  # lit test.csv

    X_train = train_df["text"]  # textes d'entraînement
    y_train = train_df["label"]  # labels d'entraînement
    X_test = test_df["text"]  # textes de test
    y_test = test_df["label"]  # labels de test

    # --- Pipeline TF-IDF + Logistic Regression ---
    pipeline = Pipeline(  # on crée une chaîne de traitement
        steps=[
            ("tfidf", TfidfVectorizer()),  # étape 1 : vectorisation TF-IDF
            ("clf", LogisticRegression(max_iter=1000))  # étape 2 : classifieur logreg
        ]
    )

    # --- Grille d'hyperparamètres à tester ---
    param_grid = {
        "tfidf__max_features": [20000, 40000],  # taille du vocabulaire
        "tfidf__ngram_range": [(1, 1), (1, 2)],  # unigrams vs unigrams+bigrams
        "tfidf__min_df": [2, 5],  # ignorer mots trop rares
        "clf__C": [0.5, 1.0, 2.0],  # régularisation logreg
        "clf__class_weight": [None, "balanced"]  # gérer déséquilibre léger
    }

    # --- Recherche des meilleurs paramètres ---
    grid = GridSearchCV(  # outil de recherche
        estimator=pipeline,  # pipeline à optimiser
        param_grid=param_grid,  # grille ci-dessus
        cv=3,  # validation croisée 3 folds
        scoring="f1_macro",  # métrique d’optimisation (bonne pour multi-classes)
        n_jobs=-1,  # utilise tous les coeurs CPU
        verbose=2  # affiche la progression
    )

    print(" Lancement GridSearch...")  # log
    grid.fit(X_train, y_train)  # entraîne et teste toutes les combinaisons

    best_model = grid.best_estimator_  # récupère le meilleur pipeline
    print(" Meilleurs paramètres :", grid.best_params_)  # affiche best params

    # --- Évaluation sur test set ---
    y_pred = best_model.predict(X_test)  # prédictions sur test

    acc = accuracy_score(y_test, y_pred)  # calcule accuracy
    print("\n Accuracy test :", acc)  # affiche accuracy

    print("\n Classification report :")  # log
    print(classification_report(y_test, y_pred))  # affiche précision/recall/F1 par classe

    print("\n Confusion matrix :")  # log
    print(confusion_matrix(y_test, y_pred))  # affiche matrice de confusion

    # --- Sauvegarde du modèle final ---
    MODEL_DIR.mkdir(parents=True, exist_ok=True)  # crée models/ si absent
    joblib.dump(best_model, MODEL_PATH)  # sauvegarde pipeline complet
    print("\n Modèle sauvegardé dans :", MODEL_PATH)  # confirmation

if __name__ == "__main__":
    train_model()  # lance l'entraînement si script exécuté directement
