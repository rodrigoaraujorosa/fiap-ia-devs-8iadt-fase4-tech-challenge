"""
Testes do pipeline de vídeo usando keypoints sintéticos (sem OpenPose nem vídeo real).

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


def test_overlay_renders_video(tmp_path):
    """render_overlay deve produzir um vídeo com o mesmo nº de frames dos JSONs."""
    cv2 = __import__("cv2")
    from src.video.anomaly import detect_anomalies
    from src.video.overlay import render_overlay
    from src.video.posture import compute_angles

    jdir = tmp_path / "json"
    jdir.mkdir()
    _make_dataset(jdir)

    # vídeo sintético (frames pretos) com o mesmo nº de frames dos JSONs
    vid = str(tmp_path / "fake.mp4")
    w = h = 200
    writer = cv2.VideoWriter(vid, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    for _ in range(_N_FRAMES):
        writer.write(np.zeros((h, w, 3), dtype=np.uint8))
    writer.release()

    kp = load_keypoints_dir(str(jdir))
    res = detect_anomalies(compute_angles(kp, fps=30.0))
    out = str(tmp_path / "overlay.mp4")
    render_overlay(vid, str(jdir), out, res=res, fps=30.0)

    import os
    assert os.path.exists(out) and os.path.getsize(out) > 0
    cap = cv2.VideoCapture(out)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert n == _N_FRAMES


def test_report_structure(tmp_path):
    """O relatório deve ter Gráfico → Análise no topo, coluna PT e tabela bem formada."""
    from src.video.anomaly import detect_anomalies
    from src.video.keypoints import coverage
    from src.video.posture import compute_angles
    from src.video.report import generate_report, plot_angles

    jdir = tmp_path / "json"
    jdir.mkdir()
    _make_dataset(jdir)
    kp = load_keypoints_dir(str(jdir))
    res = detect_anomalies(compute_angles(kp, fps=30.0))
    fig = str(tmp_path / "fig.png")
    plot_angles(res, fig, title="teste")
    path = generate_report(res, coverage(kp), str(tmp_path), "SYN", fps=30.0, fig_path=fig)

    txt = open(path, encoding="utf-8").read()
    # ordem: gráfico antes da análise
    assert txt.index("## Gráfico") < txt.index("## Análise") < txt.index("## Estatística")
    # coluna em português na estatística
    assert "Articulação" in txt and "Joelho D" in txt
    # tabela de eventos sem o cabeçalho quebrado |z|
    assert "Severidade (z máx)" in txt and "|z| máx" not in txt
    # eventos ficam no final
    assert txt.index("## Estatística") < txt.index("## Resumo e principais eventos")


def test_validate_separates_correct_incorrect():
    """validate() deve acusar mais anomalia nas repetições incorretas."""
    import pandas as pd

    from src.video.validate import validate

    # 20 frames: rep 1 (0-9, correta) sem anomalia; rep 2 (10-19, incorreta) toda anômala
    res = pd.DataFrame({"is_anomaly": [0] * 10 + [1] * 10}, index=range(20))
    seg = pd.DataFrame([
        {"video_id": "X", "repetition_number": 1, "exercise_id": 6,
         "first_frame": 0, "last_frame": 9, "correctness": 1},
        {"video_id": "X", "repetition_number": 2, "exercise_id": 6,
         "first_frame": 10, "last_frame": 19, "correctness": 0},
    ])
    per_rep, by_class = validate(res, seg, frame_step=1)
    assert len(per_rep) == 2
    assert by_class[0] > by_class[1]  # incorreta acumula mais desvios


def test_validate_frame_step_mapping():
    """Com frame_step=3, o índice i do keypoint mapeia para o frame original 3*i."""
    import pandas as pd

    from src.video.validate import label_frames

    res = pd.DataFrame({"is_anomaly": [1] * 10}, index=range(10))  # índices 0..9 -> frames 0,3,..,27
    seg = pd.DataFrame([{"video_id": "X", "repetition_number": 1, "exercise_id": 6,
                         "first_frame": 12, "last_frame": 21, "correctness": 0}])
    labeled = label_frames(res, seg, frame_step=3)
    # frames originais 12..21 -> índices 4,5,6,7 (12,15,18,21)
    inside = labeled.dropna(subset=["correctness"])
    assert list(inside.index) == [4, 5, 6, 7]


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
