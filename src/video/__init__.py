"""Entrega 1 — Análise de Vídeo (OpenPose): parser, postura, anomalia, relatório."""
from .anomaly import detect_anomalies
from .keypoints import BODY_25, coverage, load_keypoints_dir
from .posture import compute_angles
from .report import generate_report, plot_angles

__all__ = [
    "BODY_25",
    "load_keypoints_dir",
    "coverage",
    "compute_angles",
    "detect_anomalies",
    "plot_angles",
    "generate_report",
]
