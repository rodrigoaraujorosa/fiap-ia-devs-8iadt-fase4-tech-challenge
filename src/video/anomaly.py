"""
Detecção de desvios posturais (frames/eventos fora do padrão).

Combina duas visões complementares sobre a série de ângulos:

1. **Z-score robusto por ângulo** (mediana + MAD): sinaliza um frame quando
   *algum* ângulo se afasta muito do seu comportamento típico no vídeo.
   Interpretável — sabemos qual articulação desviou.
2. **IsolationForest multivariado**: aprende o padrão conjunto dos ângulos e
   marca frames globalmente atípicos (mesma técnica da Entrega 3).

Um frame é anomalia se qualquer um dos dois o sinaliza.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .posture import angle_columns

_MAD_TO_STD = 0.6745  # constante que torna o MAD comparável ao desvio-padrão


def robust_z(series: pd.Series) -> pd.Series:
    """Z-score robusto (baseado em mediana e MAD) de uma série de ângulo."""
    med = series.median()
    mad = (series - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return pd.Series(0.0, index=series.index)
    return _MAD_TO_STD * (series - med) / mad


def detect_anomalies(
    angles: pd.DataFrame,
    z_thresh: float = 3.5,
    contamination: float = 0.03,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Marca frames anômalos na série de ângulos.

    Retorna o DataFrame de entrada acrescido de:
      - ``z_anomaly``   (0/1) — sinalizado pelo z-score robusto
      - ``iso_anomaly`` (0/1) — sinalizado pelo IsolationForest
      - ``is_anomaly``  (0/1) — união dos dois
      - ``worst_angle`` — ângulo de maior desvio |z| naquele frame
      - ``anomaly_score`` — |z| máximo do frame (severidade)
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.impute import SimpleImputer

    feat_cols = angle_columns(angles)
    z = angles[feat_cols].apply(robust_z)
    z_abs = z.abs()

    res = angles.copy()
    res["z_anomaly"] = (z_abs > z_thresh).any(axis=1).astype(int)

    # IsolationForest precisa de matriz sem NaN -> imputa pela mediana da coluna.
    X = SimpleImputer(strategy="median").fit_transform(angles[feat_cols])
    iso = IsolationForest(contamination=contamination, random_state=random_state)
    res["iso_anomaly"] = (iso.fit_predict(X) == -1).astype(int)

    res["is_anomaly"] = ((res["z_anomaly"] == 1) | (res["iso_anomaly"] == 1)).astype(int)
    res["worst_angle"] = z_abs.fillna(0).idxmax(axis=1)
    res["anomaly_score"] = z_abs.max(axis=1).round(2)
    return res
