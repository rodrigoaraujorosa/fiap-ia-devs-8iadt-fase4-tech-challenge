"""
Wrapper para invocar o binário do OpenPose (Portable Demo) sobre um vídeo.

Não usamos a API Python do OpenPose (que exige compilação em C++/CUDA). Em vez
disso chamamos o executável ``OpenPoseDemo`` via linha de comando, pedindo que
ele grave os keypoints em JSON — que depois processamos com :mod:`keypoints`.

Requer o OpenPose já baixado (ver ``docs/openpose_setup.md``). Em máquinas com
GPU fraca, reduza ``net_resolution`` (ex.: "320x176") para não estourar a VRAM.
"""
from __future__ import annotations

import os
import platform
import subprocess


def _demo_binary(openpose_root: str) -> str:
    """Caminho do executável OpenPoseDemo conforme o sistema operacional."""
    if platform.system() == "Windows":
        return os.path.join(openpose_root, "bin", "OpenPoseDemo.exe")
    return os.path.join(openpose_root, "build", "examples", "openpose", "openpose.bin")


def run_openpose(
    video_path: str,
    openpose_root: str,
    out_json_dir: str,
    net_resolution: str = "320x176",
    model_pose: str = "BODY_25",
    render: bool = False,
) -> str:
    """
    Roda o OpenPose sobre ``video_path`` e grava um JSON por frame em ``out_json_dir``.

    O OpenPose precisa ser executado a partir da sua própria raiz (para achar a
    pasta ``models/``), por isso usamos ``cwd=openpose_root``. Retorna
    ``out_json_dir``.
    """
    binary = _demo_binary(openpose_root)
    if not os.path.exists(binary):
        raise FileNotFoundError(
            f"OpenPoseDemo não encontrado em {binary}. "
            "Confira o caminho do OpenPose (ver docs/openpose_setup.md)."
        )
    os.makedirs(out_json_dir, exist_ok=True)

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
    subprocess.run(cmd, cwd=openpose_root, check=True)
    return out_json_dir
