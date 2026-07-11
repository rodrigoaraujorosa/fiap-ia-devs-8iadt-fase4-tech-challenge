"""
Loader + baseline de deteccao de anomalia — UCI HAR (Human Activity Recognition).
Usado para modelar "padrao de movimentacao do paciente" e sinalizar desvios.

Uso:
    python load_uci_har.py --data "./uci_har/UCI HAR Dataset"

Requer: pandas numpy scikit-learn
"""
import argparse
import os
import numpy as np
import pandas as pd


def load_split(root: str, split: str):
    """Carrega X, y, subject de um split (train/test)."""
    feat = pd.read_csv(os.path.join(root, "features.txt"),
                       sep=r"\s+", header=None, names=["idx", "name"])
    # nomes de features tem duplicatas -> torna unicos (name, name_1, name_2, ...)
    seen: dict[str, int] = {}
    names = []
    for n in feat["name"]:
        if n in seen:
            seen[n] += 1
            names.append(f"{n}_{seen[n]}")
        else:
            seen[n] = 0
            names.append(n)
    X = pd.read_csv(os.path.join(root, split, f"X_{split}.txt"),
                    sep=r"\s+", header=None)
    X.columns = names
    y = pd.read_csv(os.path.join(root, split, f"y_{split}.txt"),
                    header=None, names=["activity_id"])
    subj = pd.read_csv(os.path.join(root, split, f"subject_{split}.txt"),
                       header=None, names=["subject"])
    labels = pd.read_csv(os.path.join(root, "activity_labels.txt"),
                         sep=r"\s+", header=None, names=["activity_id", "activity"])
    y = y.merge(labels, on="activity_id", how="left")
    return X, y, subj


def summarize(X, y, subj):
    print("\n=== Estrutura (train) ===")
    print(f"Amostras: {len(X):,}  |  Features: {X.shape[1]}")
    print(f"Sujeitos: {subj['subject'].nunique()}")
    print("\n=== Distribuicao de atividades ===")
    print(y["activity"].value_counts())


def detect_anomalies(X_train, X_test, normal_activities, y_train, y_test):
    """
    Treina IsolationForest so nas atividades 'normais' (ex.: repouso) e
    sinaliza como anomalia amostras que fogem desse padrao.
    """
    from sklearn.ensemble import IsolationForest

    mask = y_train["activity"].isin(normal_activities)
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X_train[mask])

    pred = model.predict(X_test)              # -1 anomalia, 1 normal
    res = y_test.copy()
    res["is_anomaly"] = (pred == -1).astype(int)
    print(f"\nTreino do modelo em: {normal_activities}")
    print(f"Anomalias sinalizadas no teste: {res['is_anomaly'].mean():.2%}")
    print("\nTaxa de anomalia por atividade real (teste):")
    print(res.groupby("activity")["is_anomaly"].mean().sort_values(ascending=False)
          .round(3))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help='pasta "UCI HAR Dataset"')
    args = ap.parse_args()

    X_train, y_train, subj_train = load_split(args.data, "train")
    X_test, y_test, subj_test = load_split(args.data, "test")
    summarize(X_train, y_train, subj_train)

    # cenario internacao: repouso e o normal; movimento intenso = anomalia
    detect_anomalies(X_train, X_test,
                     normal_activities=["LAYING", "SITTING", "STANDING"],
                     y_train=y_train, y_test=y_test)


if __name__ == "__main__":
    main()
