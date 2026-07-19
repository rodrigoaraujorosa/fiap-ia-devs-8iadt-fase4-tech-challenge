"""
Padrões de movimentação do paciente — UCI HAR (Human Activity Recognition).

Enquadramento clínico: durante a internação, o esperado é o paciente em **repouso**
(deitado, sentado ou em pé parado). Marcha — sobretudo subir e descer escada — é
movimentação inesperada em leito e deve gerar alerta.

O modelo é treinado **só com as classes de repouso** e nunca vê marcha no treino; as
atividades de movimento funcionam como ground-truth de anomalia no teste. Isso mantém a
detecção não-supervisionada, como nas Entregas 1 e 2, mas permite medir precisão e recall
de verdade em vez de apenas exibir a taxa de detecção.

Módulo de biblioteca: o ponto de entrada é ``python -m src.anomaly.cli``.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

# Em leito, estas são as atividades esperadas; as demais (marcha) são o alvo do alerta.
REST_ACTIVITIES = ["LAYING", "SITTING", "STANDING"]

RANDOM_STATE = 42


def load_split(root: str, split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carrega X (561 features), y (atividade) e o id do sujeito de um split."""
    feat = pd.read_csv(os.path.join(root, "features.txt"),
                       sep=r"\s+", header=None, names=["idx", "name"])
    # os nomes de feature do dataset têm duplicatas — torna únicos (name, name_1, ...)
    seen: dict[str, int] = {}
    names = []
    for n in feat["name"]:
        if n in seen:
            seen[n] += 1
            names.append(f"{n}_{seen[n]}")
        else:
            seen[n] = 0
            names.append(n)

    X = pd.read_csv(os.path.join(root, split, f"X_{split}.txt"), sep=r"\s+", header=None)
    X.columns = names
    y = pd.read_csv(os.path.join(root, split, f"y_{split}.txt"),
                    header=None, names=["activity_id"])
    subject = pd.read_csv(os.path.join(root, split, f"subject_{split}.txt"),
                          header=None, names=["subject"])
    labels = pd.read_csv(os.path.join(root, "activity_labels.txt"),
                         sep=r"\s+", header=None, names=["activity_id", "activity"])
    y = y.merge(labels, on="activity_id", how="left")
    return X, y, subject


def detect(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    contamination: float = 0.05,
    rest_activities: list[str] | None = None,
) -> pd.DataFrame:
    """
    Treina o IsolationForest apenas nas atividades de repouso e classifica o teste.

    Devolve o y_test acrescido de ``is_anomaly`` (1 = alerta) e ``is_movement``
    (ground-truth: 1 quando a atividade real não é de repouso).
    """
    from sklearn.ensemble import IsolationForest

    rest = rest_activities or REST_ACTIVITIES
    mask = y_train["activity"].isin(rest)

    model = IsolationForest(contamination=contamination, random_state=RANDOM_STATE)
    model.fit(X_train[mask])

    res = y_test.copy()
    res["is_anomaly"] = (model.predict(X_test) == -1).astype(int)
    res["is_movement"] = (~res["activity"].isin(rest)).astype(int)
    # score bruto: quanto menor, mais anômalo — usado para a curva ROC
    res["score"] = model.score_samples(X_test)
    return res


def evaluate(res: pd.DataFrame) -> dict:
    """
    Métricas do alerta contra o ground-truth de atividade.

    Classe positiva = movimentação inesperada (marcha). O AUC usa o score contínuo do
    modelo, então não depende do limiar escolhido pela ``contamination``.
    """
    from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                                 roc_auc_score)

    y_true = res["is_movement"]
    y_pred = res["is_anomaly"]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "samples": len(res),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        # score menor = mais anômalo, então inverte o sinal para o AUC
        "roc_auc": float(roc_auc_score(y_true, -res["score"])),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "alert_rate": float(y_pred.mean()),
        "false_alarm_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
    }


def per_activity(res: pd.DataFrame) -> pd.DataFrame:
    """Taxa de alerta por atividade real — mostra onde o modelo erra."""
    out = (res.groupby("activity")
              .agg(samples=("is_anomaly", "size"), alert_rate=("is_anomaly", "mean"))
              .sort_values("alert_rate", ascending=False))
    out["is_movement"] = ~out.index.isin(REST_ACTIVITIES)
    return out.round(4)


def run(data_root: str, contamination: float = 0.05) -> dict:
    """Carrega, detecta e avalia. Devolve tudo o que o relatório precisa."""
    X_train, y_train, subj_train = load_split(data_root, "train")
    X_test, y_test, subj_test = load_split(data_root, "test")

    res = detect(X_train, y_train, X_test, y_test, contamination=contamination)
    metrics = evaluate(res)
    metrics.update({
        "train_samples": len(X_train),
        "train_rest_samples": int(y_train["activity"].isin(REST_ACTIVITIES).sum()),
        "features": X_train.shape[1],
        "subjects_train": int(subj_train["subject"].nunique()),
        "subjects_test": int(subj_test["subject"].nunique()),
        "contamination": contamination,
    })
    return {"results": res, "metrics": metrics, "per_activity": per_activity(res)}


def plot(res: pd.DataFrame, out_path: str) -> str:
    """Figura: taxa de alerta por atividade, separando repouso de marcha."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tab = per_activity(res).sort_values("alert_rate")
    cores = ["#c0392b" if mov else "#2e86c1" for mov in tab["is_movement"]]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(tab.index, tab["alert_rate"] * 100, color=cores)
    ax.set_xlabel("frames sinalizados como anomalia (%)")
    ax.set_title("Alerta de movimentação por atividade real — UCI HAR")
    ax.set_xlim(0, 105)
    for i, v in enumerate(tab["alert_rate"] * 100):
        ax.text(v + 1.5, i, f"{v:.1f}%", va="center", fontsize=9)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#2e86c1", label="repouso (esperado em leito)"),
                       Patch(color="#c0392b", label="marcha (deve alertar)")],
              loc="lower right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
