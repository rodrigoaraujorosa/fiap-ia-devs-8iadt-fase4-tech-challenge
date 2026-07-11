"""
Testes do pipeline de vídeo usando keypoints sintéticos (sem OpenPose/KIMORE).

Gera JSONs no formato do OpenPose: um esqueleto em pé "normal" com ruído leve e,
no meio, alguns frames com o joelho direito bruscamente flexionado (anomalia).
Valida o parser, o cálculo de ângulos e a detecção de desvios.
"""
import json
import math

import numpy as np

from src.video import (
    compute_angles,
    detect_anomalies,
    load_keypoints_dir,
)
from src.video.keypoints import BODY_25
from src.video.posture import _angle

# Esqueleto de referência em pé (coordenadas de imagem, y cresce para baixo).
_BASE = {
    "Nose": (100, 50), "Neck": (100, 80),
    "RShoulder": (80, 85), "RElbow": (75, 120), "RWrist": (72, 155),
    "LShoulder": (120, 85), "LElbow": (125, 120), "LWrist": (128, 155),
    "MidHip": (100, 150), "RHip": (88, 150), "RKnee": (88, 200), "RAnkle": (88, 250),
    "LHip": (112, 150), "LKnee": (112, 200), "LAnkle": (112, 250),
    "REye": (95, 45), "LEye": (105, 45), "REar": (90, 50), "LEar": (110, 50),
    "LBigToe": (118, 260), "LSmallToe": (122, 260), "LHeel": (112, 258),
    "RBigToe": (82, 260), "RSmallToe": (78, 260), "RHeel": (88, 258),
}

_N_FRAMES = 100
_ANOMALY_FRAMES = {40, 41, 42, 43, 44}


def _write_frame(path, coords, conf=0.8):
    kp = []
    for name in BODY_25:
        x, y = coords[name]
        kp += [float(x), float(y), conf]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"version": 1.3, "people": [{"pose_keypoints_2d": kp}]}, fh)


def _make_dataset(dir_path, seed=0):
    rng = np.random.default_rng(seed)
    for i in range(_N_FRAMES):
        coords = {k: (x + rng.normal(0, 0.5), y + rng.normal(0, 0.5))
                  for k, (x, y) in _BASE.items()}
        if i in _ANOMALY_FRAMES:
            # joelho direito bruscamente flexionado: tornozelo vai para a frente
            coords["RAnkle"] = (132 + rng.normal(0, 0.5), 214 + rng.normal(0, 0.5))
        _write_frame(dir_path / f"synt_{i:012d}_keypoints.json", coords)


def test_angle_right_angle():
    """_angle deve medir ~90° num canto reto."""
    a = np.array([[1.0, 0.0]])   # b->a horizontal
    b = np.array([[0.0, 0.0]])   # vértice
    c = np.array([[0.0, 1.0]])   # b->c vertical
    assert math.isclose(_angle(a, b, c)[0], 90.0, abs_tol=1e-6)


def test_pipeline_flags_injected_anomalies(tmp_path):
    _make_dataset(tmp_path)

    kp = load_keypoints_dir(str(tmp_path))
    assert len(kp) == _N_FRAMES
    # joelho reto no frame 0 -> ângulo perto de 180°
    angles = compute_angles(kp, fps=30.0)
    assert angles.loc[0, "r_knee"] > 165

    res = detect_anomalies(angles)

    flagged = set(res.index[res["is_anomaly"] == 1])
    # todos os frames injetados devem ser pegos...
    assert _ANOMALY_FRAMES <= flagged
    # ...e o joelho direito deve ser o ângulo predominante neles
    assert (res.loc[sorted(_ANOMALY_FRAMES), "worst_angle"] == "r_knee").all()
    # sem excesso de falsos positivos nos frames normais
    normal = flagged - _ANOMALY_FRAMES
    assert len(normal) <= 5
