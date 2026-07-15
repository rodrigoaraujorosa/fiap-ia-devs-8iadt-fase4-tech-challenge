"""
Wrapper para invocar o binário do OpenPose (Portable Demo) sobre um vídeo.

Não usamos a API Python do OpenPose (que exige compilação em C++/CUDA). Em vez
disso chamamos o executável ``OpenPoseDemo`` via linha de comando, pedindo que
ele grave os keypoints em JSON — que depois processamos com :mod:`keypoints`.

O OpenPose não reporta progresso, mas escreve **um JSON por frame**. Então
acompanhamos o andamento **contando os JSONs** na pasta de saída enquanto o
processo roda, e repassamos ``(feito, total)`` a um ``progress_cb`` — usado tanto
pela barra no terminal quanto pela app web.

Requer o OpenPose já baixado (ver ``docs/openpose_setup.md``). Em máquinas com
GPU fraca, reduza ``net_resolution`` (ex.: "320x176") para não estourar a VRAM.
"""
from __future__ import annotations

import glob
import os
import platform
import subprocess
import time
from collections.abc import Callable

ProgressCb = Callable[[int, int], None]


def _demo_binary(openpose_root: str) -> str:
    """Caminho do executável OpenPoseDemo conforme o sistema operacional."""
    if platform.system() == "Windows":
        return os.path.join(openpose_root, "bin", "OpenPoseDemo.exe")
    return os.path.join(openpose_root, "build", "examples", "openpose", "openpose.bin")


def count_json(out_dir: str) -> int:
    """Quantos ``*_keypoints.json`` já foram escritos na pasta."""
    return len(glob.glob(os.path.join(out_dir, "*_keypoints.json")))


def video_frame_count(video_path: str) -> int | None:
    """Número de frames do vídeo (via OpenCV), ou None se indisponível."""
    try:
        import cv2

        cap = cv2.VideoCapture(video_path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n if n > 0 else None
    except Exception:
        return None


def run_openpose(
    video_path: str,
    openpose_root: str,
    out_json_dir: str,
    net_resolution: str = "320x176",
    model_pose: str = "BODY_25",
    render: bool = False,
    progress_cb: ProgressCb | None = None,
    poll: float = 0.5,
) -> str:
    """
    Roda o OpenPose sobre ``video_path`` e grava um JSON por frame em ``out_json_dir``.

    O OpenPose precisa ser executado a partir da sua própria raiz (para achar a
    pasta ``models/``), por isso usamos ``cwd=openpose_root``. Se ``progress_cb`` for
    passado, é chamado periodicamente com ``(frames_prontos, total_frames)``.
    Retorna ``out_json_dir``.
    """
    binary = _demo_binary(openpose_root)
    if not os.path.exists(binary):
        raise FileNotFoundError(
            f"OpenPoseDemo não encontrado em {binary}. "
            "Confira o caminho do OpenPose (ver docs/openpose_setup.md)."
        )
    os.makedirs(out_json_dir, exist_ok=True)
    total = video_frame_count(os.path.abspath(video_path))

    cmd = [
        binary,
        "--video", os.path.abspath(video_path),
        "--write_json", os.path.abspath(out_json_dir),
        "--model_pose", model_pose,
        "--net_resolution", net_resolution,
        "--display", "0",
        "--render_pose", "1" if render else "0",
    ]
    print("Executando OpenPose:\n  " + " ".join(cmd))

    # OpenPose é verboso; jogamos a saída num log para não poluir a barra de progresso.
    log_path = os.path.join(out_json_dir, "_openpose.log")
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(cmd, cwd=openpose_root, stdout=log,
                                stderr=subprocess.STDOUT)
        while proc.poll() is None:
            if progress_cb and total:
                progress_cb(min(count_json(out_json_dir), total), total)
            time.sleep(poll)
        rc = proc.wait()

    if progress_cb and total:
        progress_cb(total, total)
    if rc != 0:
        tail = _tail(log_path)
        raise RuntimeError(f"OpenPose falhou (código {rc}). Últimas linhas do log:\n{tail}")
    return out_json_dir


def _tail(path: str, n: int = 15) -> str:
    """Últimas ``n`` linhas de um arquivo (para mensagens de erro)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return "".join(fh.readlines()[-n:])
    except OSError:
        return "(log indisponível)"
