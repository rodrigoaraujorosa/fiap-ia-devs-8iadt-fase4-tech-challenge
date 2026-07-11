"""
Validação da detecção de desvios contra o ground-truth do REHAB24-6.

O REHAB24-6 rotula cada *repetição* de exercício com `correctness` (1=correta,
0=incorreta) e o intervalo de frames (`first_frame`..`last_frame`). Aqui cruzamos
esses rótulos com os frames que o nosso detector sinalizou como anomalia, para
responder: *repetições incorretas concentram mais frames anômalos que as corretas?*

Uso típico (após rodar o pipeline e ter o DataFrame `res` de anomaly.detect_anomalies):

    seg = load_segmentation("data/video/rehab24-6/Segmentation.csv", "PM_008")
    per_rep, by_class = validate(res, seg)
"""
from __future__ import annotations

import pandas as pd

# id -> nome do exercício (conforme documentação do REHAB24-6)
EXERCISES = {
    1: "Abdução de braço", 2: "Arm VW", 3: "Flexões",
    4: "Abdução de perna", 5: "Lunge", 6: "Agachamento",
}


def load_segmentation(csv_path: str, video_id: str | None = None) -> pd.DataFrame:
    """Carrega o Segmentation.csv (separado por ';'), opcionalmente filtrando um vídeo."""
    df = pd.read_csv(csv_path, sep=";")
    if video_id is not None:
        df = df[df["video_id"] == video_id].copy()
        if df.empty:
            raise ValueError(f"video_id '{video_id}' não encontrado no Segmentation.csv")
    return df


def label_frames(res: pd.DataFrame, seg: pd.DataFrame) -> pd.DataFrame:
    """
    Anota cada frame de ``res`` com ``repetition``, ``exercise_id`` e ``correctness``,
    segundo os intervalos de frame do ``seg``. Frames fora de qualquer repetição
    (ex.: preparação entre séries) ficam como ``NA``.
    """
    out = res.copy()
    out["repetition"] = pd.NA
    out["exercise_id"] = pd.NA
    out["correctness"] = pd.NA
    for _, row in seg.iterrows():
        mask = (out.index >= row["first_frame"]) & (out.index <= row["last_frame"])
        out.loc[mask, "repetition"] = row["repetition_number"]
        out.loc[mask, "exercise_id"] = row["exercise_id"]
        out.loc[mask, "correctness"] = row["correctness"]
    return out


def validate(res: pd.DataFrame, seg: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Retorna (per_rep, by_class):
      - ``per_rep``: taxa de frames anômalos por repetição (com sua correctness);
      - ``by_class``: taxa média de anomalia agregada por correta (1) vs incorreta (0).

    A expectativa (a validar com os dados) é ``by_class[0] > by_class[1]``: repetições
    incorretas devem acumular mais frames sinalizados.
    """
    labeled = label_frames(res, seg)
    inside = labeled.dropna(subset=["correctness"])
    per_rep = (
        inside.groupby(["repetition", "exercise_id", "correctness"])["is_anomaly"]
        .mean()
        .reset_index()
        .rename(columns={"is_anomaly": "anomaly_rate"})
    )
    by_class = per_rep.groupby("correctness")["anomaly_rate"].mean()
    return per_rep, by_class
