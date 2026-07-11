"""
Cálculo de ângulos articulares a partir dos keypoints (análise postural).

Converte as coordenadas 2D das juntas (BODY_25) em uma série temporal de
ângulos clinicamente relevantes para os exercícios de reabilitação física do
REHAB24-6 (agachamento, lunge, abdução de braço/perna, etc.). Esses ângulos são a base
para detectar desvios posturais.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .keypoints import joint_xy

# Ângulos de 3 pontos: (junta_A, VÉRTICE, junta_C) -> ângulo no vértice.
JOINT_ANGLES: dict[str, tuple[str, str, str]] = {
    "r_elbow": ("RShoulder", "RElbow", "RWrist"),
    "l_elbow": ("LShoulder", "LElbow", "LWrist"),
    "r_shoulder": ("RElbow", "RShoulder", "RHip"),
    "l_shoulder": ("LElbow", "LShoulder", "LHip"),
    "r_hip": ("RShoulder", "RHip", "RKnee"),
    "l_hip": ("LShoulder", "LHip", "LKnee"),
    "r_knee": ("RHip", "RKnee", "RAnkle"),
    "l_knee": ("LHip", "LKnee", "LAnkle"),
}


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Ângulo (graus) no vértice ``b`` entre os segmentos b->a e b->c.

    Cada argumento é (n_frames, 2). Frames com qualquer ponto ``NaN`` resultam
    em ``NaN`` (junta não detectada com confiança).
    """
    v1 = a - b
    v2 = c - b
    denom = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (v1 * v2).sum(-1) / denom
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _inclination_from_vertical(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    """
    Inclinação (graus) do segmento bottom->top em relação à vertical da imagem.

    0° = perfeitamente ereto (paciente em pé/alinhado); valores maiores = tronco
    mais inclinado. Na imagem o eixo Y cresce para baixo, então "para cima" é
    o vetor (0, -1).
    """
    v = top - bottom
    vertical = np.array([0.0, -1.0])
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (v @ vertical) / np.linalg.norm(v, axis=-1)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def compute_angles(df: pd.DataFrame, fps: float = 30.0) -> pd.DataFrame:
    """
    Gera a série temporal de ângulos posturais a partir do DataFrame de keypoints.

    Retorna um DataFrame indexado por frame, com uma coluna ``time_s`` (segundos)
    e uma coluna por ângulo definido em :data:`JOINT_ANGLES` mais
    ``trunk_inclination``.
    """
    out = pd.DataFrame(index=df.index)
    out["time_s"] = df.index.to_numpy(dtype=float) / fps

    for name, (a, b, c) in JOINT_ANGLES.items():
        out[name] = _angle(joint_xy(df, a), joint_xy(df, b), joint_xy(df, c))

    out["trunk_inclination"] = _inclination_from_vertical(
        joint_xy(df, "Neck"), joint_xy(df, "MidHip")
    )
    return out


def angle_columns(angles: pd.DataFrame) -> list[str]:
    """Nomes das colunas de ângulo (exclui a coluna auxiliar ``time_s``)."""
    return [c for c in angles.columns if c != "time_s"]
