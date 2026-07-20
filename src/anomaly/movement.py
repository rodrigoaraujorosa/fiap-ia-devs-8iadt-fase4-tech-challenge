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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_PATH = Path("models/movement_detector.joblib")

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


@dataclass
class MovementDetector:
    """Detector treinado em repouso, pronto para classificar leituras de um sujeito."""
    model: object
    rest_activities: list[str]
    contamination: float
    trained_on: dict

    def score(self, X: pd.DataFrame, y: pd.DataFrame | None = None) -> pd.DataFrame:
        """Classifica leituras. ``y`` é opcional e serve só para conferência posterior."""
        res = y.copy() if y is not None else pd.DataFrame(index=X.index)
        res["is_anomaly"] = (self.model.predict(X) == -1).astype(int)
        res["score"] = self.model.score_samples(X)
        if y is not None:
            res["is_movement"] = (~res["activity"].isin(self.rest_activities)).astype(int)
        return res


def fit(X_train: pd.DataFrame, y_train: pd.DataFrame, contamination: float = 0.05,
        rest_activities: list[str] | None = None) -> MovementDetector:
    """Treina o IsolationForest **apenas** nas atividades de repouso."""
    from sklearn.ensemble import IsolationForest

    rest = rest_activities or REST_ACTIVITIES
    mask = y_train["activity"].isin(rest)

    model = IsolationForest(contamination=contamination, random_state=RANDOM_STATE)
    model.fit(X_train[mask])

    return MovementDetector(
        model=model,
        rest_activities=rest,
        contamination=contamination,
        trained_on={"samples": int(mask.sum()), "features": X_train.shape[1]},
    )


def save(detector: MovementDetector, path: Path = MODEL_PATH) -> Path:
    import joblib
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(detector, path)
    return path


def load(path: Path = MODEL_PATH) -> MovementDetector:
    import joblib
    if not path.exists():
        raise FileNotFoundError(
            f"nenhum modelo em {path} — treine antes: python -m src.anomaly.cli --train")
    return joblib.load(path)


def monitor_subject(subject_id: int, data_root: str,
                    detector: MovementDetector | None = None) -> dict:
    """
    Modo de inferência: classifica as leituras de **um** sujeito do conjunto de teste.

    O sujeito não participou do treino. Devolve as janelas em que o alerta dispararia e,
    em separado, a atividade real de cada uma, para conferência.
    """
    det = detector or load()
    X, y, subj = load_split(data_root, "test")

    mask = subj["subject"] == subject_id
    if not mask.any():
        disponiveis = sorted(subj["subject"].unique().tolist())
        raise ValueError(f"sujeito {subject_id} não está no teste; disponíveis: {disponiveis}")

    res = det.score(X[mask], y[mask])
    res["window"] = np.arange(len(res))

    alertas = res[res["is_anomaly"] == 1]
    return {
        "data": res,
        "summary": {
            "subject": int(subject_id),
            "windows": int(len(res)),
            "alerts": int(len(alertas)),
            "alert_rate": float(res["is_anomaly"].mean()),
            "recall_movement": float(
                res.loc[res["is_movement"] == 1, "is_anomaly"].mean())
                if (res["is_movement"] == 1).any() else None,
            "false_alarm_rate": float(
                res.loc[res["is_movement"] == 0, "is_anomaly"].mean())
                if (res["is_movement"] == 0).any() else None,
        },
    }


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

    detector = fit(X_train, y_train, contamination=contamination)
    res = detector.score(X_test, y_test)
    metrics = evaluate(res)
    metrics.update({
        "train_samples": len(X_train),
        "train_rest_samples": detector.trained_on["samples"],
        "features": X_train.shape[1],
        "subjects_train": int(subj_train["subject"].nunique()),
        "subjects_test": int(subj_test["subject"].nunique()),
        "contamination": contamination,
    })
    return {"results": res, "metrics": metrics, "per_activity": per_activity(res),
            "detector": detector,
            "test_subjects": sorted(subj_test["subject"].unique().tolist())}


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
