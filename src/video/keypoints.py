"""
Parser dos keypoints do OpenPose (formato BODY_25).

O OpenPose grava **um arquivo JSON por frame** (opção ``--write_json``), com a
lista de pessoas detectadas e, para cada uma, o vetor ``pose_keypoints_2d`` =
[x0, y0, c0, x1, y1, c1, ...] com 25 juntas (x, y em pixels e c = confiança 0-1).

Este módulo carrega essa pasta de JSONs e devolve um ``DataFrame`` com uma linha
por frame e colunas ``<Junta>_x``, ``<Junta>_y``, ``<Junta>_c``.
"""
from __future__ import annotations

import glob
import json
import math
import os

import numpy as np
import pandas as pd

# Ordem oficial das 25 juntas do modelo BODY_25 do OpenPose.
BODY_25 = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow",
    "LWrist", "MidHip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
    "REye", "LEye", "REar", "LEar", "LBigToe", "LSmallToe", "LHeel",
    "RBigToe", "RSmallToe", "RHeel",
]


def _person_metrics(kp: list[float], min_confidence: float):
    """
    Métricas de uma pessoa: (área do bounding box, centroide, confiança média),
    considerando apenas as juntas com confiança >= ``min_confidence``. Retorna
    ``None`` se houver poucas juntas confiáveis.
    """
    arr = np.asarray(kp, dtype=float).reshape(-1, 3)  # (25, 3): x, y, c
    valid = arr[arr[:, 2] >= min_confidence]
    if len(valid) < 3:
        return None
    x, y = valid[:, 0], valid[:, 1]
    area = float((x.max() - x.min()) * (y.max() - y.min()))
    centroid = (float(x.mean()), float(y.mean()))
    mean_conf = float(valid[:, 2].mean())
    return area, centroid, mean_conf


def _select_person(people: list[dict], min_confidence: float,
                   prev_centroid: tuple[float, float] | None):
    """
    Seleciona a pessoa principal do frame de forma robusta e devolve
    ``(keypoints, centroide)``.

    Vídeos clínicos podem ter observadores ao fundo. A abordagem antiga (maior
    confiança total) às vezes travava na pessoa errada. Aqui:

    - prioriza a pessoa **maior e mais confiante** (``área × confiança média``) —
      o paciente em primeiro plano domina; gente menor ao fundo é descartada;
    - **estabiliza no tempo**: entre candidatos, favorece quem está mais próximo
      da seleção do frame anterior, evitando "pular" de uma pessoa para outra.
    """
    cands = []
    for person in people:
        kp = person.get("pose_keypoints_2d", [])
        if not kp:
            continue
        m = _person_metrics(kp, min_confidence)
        if m is not None:
            cands.append((kp, *m))  # (kp, área, centroide, conf_média)
    if not cands:
        return None, prev_centroid

    def score(cand) -> float:
        _kp, area, centroid, mean_conf = cand
        s = area * mean_conf  # paciente = grande e confiante
        if prev_centroid is not None:
            d = math.hypot(centroid[0] - prev_centroid[0], centroid[1] - prev_centroid[1])
            s /= 1.0 + d / (math.sqrt(area) + 1e-6)  # continuidade temporal
        return s

    best = max(cands, key=score)
    return best[0], best[2]  # keypoints, centroide


def load_keypoints_dir(json_dir: str, min_confidence: float = 0.1) -> pd.DataFrame:
    """
    Carrega todos os ``*_keypoints.json`` de uma pasta em ordem de frame.

    Juntas com confiança abaixo de ``min_confidence`` (ou não detectadas, c=0)
    têm x/y marcados como ``NaN`` — assim os ângulos calculados a partir delas
    também ficam ``NaN`` em vez de virarem lixo em (0, 0).
    """
    files = sorted(glob.glob(os.path.join(json_dir, "*_keypoints.json")))
    if not files:
        raise FileNotFoundError(f"Nenhum *_keypoints.json encontrado em {json_dir}")

    rows = []
    prev_centroid: tuple[float, float] | None = None
    for frame_idx, path in enumerate(files):
        with open(path, encoding="utf-8") as fh:
            people = json.load(fh).get("people", [])
        kp, prev_centroid = _select_person(people, min_confidence, prev_centroid)

        row: dict[str, float] = {"frame": frame_idx}
        for j, name in enumerate(BODY_25):
            if kp is None:
                x = y = c = np.nan
            else:
                x, y, c = kp[3 * j], kp[3 * j + 1], kp[3 * j + 2]
                if c < min_confidence:
                    x = y = np.nan
            row[f"{name}_x"], row[f"{name}_y"], row[f"{name}_c"] = x, y, c
        rows.append(row)

    return pd.DataFrame(rows).set_index("frame")


def joint_xy(df: pd.DataFrame, name: str) -> np.ndarray:
    """Retorna array (n_frames, 2) com as coordenadas (x, y) de uma junta."""
    return df[[f"{name}_x", f"{name}_y"]].to_numpy(dtype=float)


def coverage(df: pd.DataFrame) -> pd.Series:
    """Fração de frames em que cada junta foi detectada (útil p/ diagnóstico)."""
    cols = [f"{name}_x" for name in BODY_25]
    cov = df[cols].notna().mean()
    cov.index = BODY_25
    return (cov * 100).round(1)
