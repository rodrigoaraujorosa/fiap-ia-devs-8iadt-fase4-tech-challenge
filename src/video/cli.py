"""
Pipeline fim-a-fim da Entrega 1 (Análise de Vídeo).

Duas formas de uso:

1. A partir de um vídeo (roda o OpenPose primeiro):
     python -m src.video.cli --video data/video/rehab24-6/PM_008-Camera17-30fps.mp4 \
         --openpose-root tools/openpose --fps 30

2. A partir de uma pasta de JSONs já extraídos (ex.: gerados no Colab):
     python -m src.video.cli --json-dir reports/json/PM_008 --fps 30

Gera o relatório e o gráfico em ``reports/``.
"""
from __future__ import annotations

import argparse
import os

from .anomaly import detect_anomalies
from .keypoints import coverage, load_keypoints_dir
from .posture import compute_angles
from .report import generate_report, plot_angles


def run_pipeline(json_dir: str, video_name: str, out_dir: str, fps: float = 30.0):
    """
    Executa parser -> ângulos -> detecção -> relatório sobre uma pasta de JSONs.
    Retorna o DataFrame ``res`` (para reuso, ex.: overlay).
    """
    print(f"[1/4] Carregando keypoints de {json_dir} ...")
    kp = load_keypoints_dir(json_dir)
    cov = coverage(kp)
    print(f"      {len(kp)} frames carregados.")

    print("[2/4] Calculando ângulos posturais ...")
    angles = compute_angles(kp, fps=fps)

    print("[3/4] Detectando desvios ...")
    res = detect_anomalies(angles)
    n_anom = int(res["is_anomaly"].sum())
    print(f"      {n_anom} frames anômalos ({n_anom / len(res):.1%}).")

    print("[4/4] Gerando relatório ...")
    fig_path = os.path.join(out_dir, "figures", f"angulos_{video_name}.png")
    plot_angles(res, fig_path, title=f"Ângulos posturais — {video_name}")
    report_path = generate_report(res, cov, out_dir, video_name, fps=fps,
                                  fig_path=fig_path)
    print(f"\nRelatório: {report_path}\nGráfico:   {fig_path}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Análise postural de vídeo clínico (OpenPose).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--json-dir", help="pasta com os *_keypoints.json já extraídos")
    src.add_argument("--video", help="vídeo a processar (roda o OpenPose antes)")
    ap.add_argument("--openpose-root", help="raiz do OpenPose (obrigatório com --video)")
    ap.add_argument("--fps", type=float, default=30.0, help="frames por segundo do vídeo")
    ap.add_argument("--out", default="reports", help="pasta de saída dos relatórios")
    ap.add_argument("--net-resolution", default="320x176",
                    help="resolução da rede do OpenPose (menor = menos VRAM)")
    ap.add_argument("--overlay", action="store_true",
                    help="também gera vídeo com esqueleto e desvios sobrepostos "
                         "(requer --video)")
    args = ap.parse_args()

    if args.video:
        if not args.openpose_root:
            ap.error("--openpose-root é obrigatório quando se usa --video")
        from .run_openpose import run_openpose
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        json_dir = os.path.join(args.out, "json", video_name)
        run_openpose(args.video, args.openpose_root, json_dir,
                     net_resolution=args.net_resolution)
    else:
        if args.overlay:
            ap.error("--overlay requer --video (precisamos do vídeo original)")
        json_dir = args.json_dir
        video_name = os.path.basename(os.path.normpath(json_dir)).replace("_json", "")

    res = run_pipeline(json_dir, video_name, args.out, fps=args.fps)

    if args.overlay:
        from .overlay import render_overlay
        overlay_path = os.path.join(args.out, f"{video_name}_overlay.mp4")
        render_overlay(args.video, json_dir, overlay_path, res=res, fps=args.fps)


if __name__ == "__main__":
    main()
