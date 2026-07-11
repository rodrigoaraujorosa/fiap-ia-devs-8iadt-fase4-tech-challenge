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


def _select_person(people: list[dict]) -> list[float] | None:
    """
    Escolhe a pessoa principal do frame = a de maior confiança total.

    Vídeos clínicos costumam ter um paciente por cena, mas o OpenPose pode
    detectar ruído/observadores ao fundo. Somamos a confiança das juntas e
    ficamos com a detecção mais forte.
    """
    best_kp, best_score = None, -1.0
    for person in people:
        kp = person.get("pose_keypoints_2d", [])
        if not kp:
            continue
        score = float(np.sum(kp[2::3]))  # confiança = todo 3º valor
        if score > best_score:
            best_kp, best_score = kp, score
    return best_kp


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
    for frame_idx, path in enumerate(files):
        with open(path, encoding="utf-8") as fh:
            people = json.load(fh).get("people", [])
        kp = _select_person(people) if people else None

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
