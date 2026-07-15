"""
Overlay do esqueleto BODY_25 sobre o vídeo, com destaque para os frames de desvio.

Desenhamos o esqueleto nós mesmos (OpenCV) a partir dos keypoints, em vez de usar o
render do OpenPose, porque assim podemos **anotar visualmente as anomalias** detectadas
pelo nosso pipeline: borda vermelha no frame, o ângulo que mais desviou e a severidade.
Gera um vídeo anotado — material direto para o vídeo-demo da entrega.

Uso:
    python -m src.video.overlay --video data/video/rehab24-6/PM_008.mp4 \
        --json-dir reports/json/PM_008 --out reports/PM_008_overlay.mp4
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from .keypoints import BODY_25, load_keypoints_dir

# Conexões (ossos) do modelo BODY_25, por índice de junta.
SKELETON = [
    (1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (8, 9), (9, 10),
    (10, 11), (8, 12), (12, 13), (13, 14), (1, 0), (0, 15), (15, 17), (0, 16),
    (16, 18), (14, 19), (19, 20), (14, 21), (11, 22), (22, 23), (11, 24),
]

# Juntas que participam de cada ângulo (para destacar o "worst_angle" em vermelho).
_ANGLE_JOINTS = {
    "r_elbow": ["RShoulder", "RElbow", "RWrist"],
    "l_elbow": ["LShoulder", "LElbow", "LWrist"],
    "r_shoulder": ["RElbow", "RShoulder", "RHip"],
    "l_shoulder": ["LElbow", "LShoulder", "LHip"],
    "r_hip": ["RShoulder", "RHip", "RKnee"],
    "l_hip": ["LShoulder", "LHip", "LKnee"],
    "r_knee": ["RHip", "RKnee", "RAnkle"],
    "l_knee": ["LHip", "LKnee", "LAnkle"],
    "trunk_inclination": ["Neck", "MidHip"],
}

_BONE_COLOR = (0, 255, 0)       # verde (BGR)
_JOINT_COLOR = (0, 220, 255)    # amarelo
_ALERT_COLOR = (0, 0, 255)      # vermelho


def _pt(row: pd.Series, name: str) -> tuple[int, int] | None:
    """Coordenada inteira (x, y) de uma junta, ou None se não detectada."""
    x, y = row[f"{name}_x"], row[f"{name}_y"]
    if pd.isna(x) or pd.isna(y):
        return None
    return int(round(x)), int(round(y))


def _draw_skeleton(frame, row, highlight: set[str] | None = None) -> None:
    import cv2

    highlight = highlight or set()
    hl_idx = {BODY_25.index(n) for n in highlight if n in BODY_25}
    for a, b in SKELETON:
        pa, pb = _pt(row, BODY_25[a]), _pt(row, BODY_25[b])
        if pa is None or pb is None:
            continue
        color = _ALERT_COLOR if (a in hl_idx and b in hl_idx) else _BONE_COLOR
        thick = 4 if (a in hl_idx and b in hl_idx) else 2
        cv2.line(frame, pa, pb, color, thick, cv2.LINE_AA)
    for j, name in enumerate(BODY_25):
        p = _pt(row, name)
        if p is None:
            continue
        color = _ALERT_COLOR if j in hl_idx else _JOINT_COLOR
        cv2.circle(frame, p, 4, color, -1, cv2.LINE_AA)


def _transcode_h264(src: str, dst: str) -> None:
    """
    Transcodifica ``src`` (mp4v do OpenCV) para **H.264** em ``dst`` — assim o vídeo
    toca no navegador (Gradio) e em players web. Usa o ffmpeg embutido do
    ``imageio-ffmpeg`` (ou o do sistema). Sem ffmpeg, mantém o mp4v original
    (ainda toca em players como o VLC).
    """
    ffmpeg = None
    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        import shutil

        ffmpeg = shutil.which("ffmpeg")

    if not ffmpeg:
        if src != dst:
            os.replace(src, dst)
        print("  (aviso: ffmpeg indisponível — overlay em mp4v; pode não tocar no navegador)")
        return

    import subprocess

    cmd = [
        ffmpeg, "-y", "-loglevel", "error", "-i", src,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # dimensões pares (exigência do yuv420p)
        "-movflags", "+faststart", dst,
    ]
    try:
        subprocess.run(cmd, check=True)
        if src != dst and os.path.exists(src):
            os.remove(src)
    except (subprocess.CalledProcessError, OSError) as exc:  # fallback: mantém o mp4v
        print(f"  (aviso: transcodificação H.264 falhou: {exc}; mantendo mp4v)")
        if src != dst and os.path.exists(src):
            os.replace(src, dst)


def render_overlay(
    video_path: str,
    json_dir: str,
    out_path: str,
    res: pd.DataFrame | None = None,
    fps: float | None = None,
    progress_cb=None,
) -> str:
    """
    Escreve um vídeo com o esqueleto sobreposto. Se ``res`` (saída de
    ``anomaly.detect_anomalies``) for passado, marca os frames de desvio.
    Se ``progress_cb`` for passado, é chamado com ``(feito, total)`` a cada frame.
    Retorna o caminho do vídeo gerado.
    """
    import cv2

    kp = load_keypoints_dir(json_dir)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo: {video_path}")

    src_fps = fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or len(kp)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    raw_path = out_path + ".raw.mp4"  # mp4v; depois transcodamos para H.264 (web)
    writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             src_fps, (w, h))

    idx, n_anom = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if progress_cb and total:
            progress_cb(idx + 1, total)
        if idx in kp.index:
            row = kp.loc[idx]
            highlight = None
            is_anom = res is not None and idx in res.index and \
                int(res.loc[idx, "is_anomaly"]) == 1
            if is_anom:
                worst = str(res.loc[idx, "worst_angle"])
                highlight = set(_ANGLE_JOINTS.get(worst, []))
            _draw_skeleton(frame, row, highlight)
            if is_anom:
                n_anom += 1
                score = float(res.loc[idx, "anomaly_score"])
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), _ALERT_COLOR, 6)
                cv2.putText(frame, f"DESVIO: {worst} (|z|={score:.1f})",
                            (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            _ALERT_COLOR, 2, cv2.LINE_AA)
        # HUD com tempo/frame
        cv2.putText(frame, f"frame {idx}  t={idx / src_fps:0.1f}s",
                    (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)
        idx += 1

    cap.release()
    writer.release()
    _transcode_h264(raw_path, out_path)
    print(f"Overlay salvo em {out_path} ({idx} frames, {n_anom} marcados como desvio)")
    return out_path


def main() -> None:
    from .anomaly import detect_anomalies
    from .posture import compute_angles

    ap = argparse.ArgumentParser(description="Overlay do esqueleto + desvios sobre o vídeo.")
    ap.add_argument("--video", required=True)
    ap.add_argument("--json-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--no-anomaly", action="store_true",
                    help="apenas o esqueleto, sem marcar desvios")
    args = ap.parse_args()

    res = None
    if not args.no_anomaly:
        kp = load_keypoints_dir(args.json_dir)
        res = detect_anomalies(compute_angles(kp, fps=args.fps or 30.0))
    render_overlay(args.video, args.json_dir, args.out, res=res, fps=args.fps)


if __name__ == "__main__":
    main()
