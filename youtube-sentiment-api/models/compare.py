import pandas as pd  # lire données
from pathlib import Path  # chemins propres
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF
from sklearn.pipeline import Pipeline  # pipeline
from sklearn.svm import LinearSVC  # SVM linéaire (baseline solide)
from sklearn.ensemble import RandomForestClassifier  # Random Forest
from sklearn.metrics import accuracy_score, classification_report  # métriques

TRAIN_PATH = Path("data/processed/train.csv")  # train set
TEST_PATH = Path("data/processed/test.csv")  # test set

def compare_models():
    train_df = pd.read_csv(TRAIN_PATH)  # charge train
    test_df = pd.read_csv(TEST_PATH)  # charge test

    X_train = train_df["text"]  # textes train
    y_train = train_df["label"]  # labels train
    X_test = test_df["text"]  # textes test
    y_test = test_df["label"]  # labels test

    # mêmes paramètres TF-IDF pour être fair dans la comparaison
    tfidf = TfidfVectorizer(
        max_features=40000,  # vocab large
        ngram_range=(1, 2),  # uni + bi-grammes
        min_df=2  # ignore mots trop rares
    )

    models = {
        "LinearSVC (SVM)": LinearSVC(),  # modèle SVM rapide et souvent top
        "RandomForest": RandomForestClassifier(
            n_estimators=200,  # nombre d'arbres
            random_state=42,  # reproductible
            n_jobs=-1  # parallélise
        )
    }

    for name, clf in models.items():
        pipe = Pipeline([
            ("tfidf", tfidf),  # vectorisation
            ("clf", clf)  # classifieur
        ])

        print(f"\n===== {name} =====")
        pipe.fit(X_train, y_train)  # entraînement
        preds = pipe.predict(X_test)  # prédictions

        print("Accuracy:", accuracy_score(y_test, preds))  # accuracy
        print(classification_report(y_test, preds))  # précision/recall/F1

if __name__ == "__main__":
    compare_models()
