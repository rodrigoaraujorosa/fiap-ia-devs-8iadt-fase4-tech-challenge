"""
App web (Gradio) para demonstração local da Entrega 1 — Análise de Vídeo.

Mostra o progresso do OpenPose (contando os JSONs) e do overlay e, ao final,
exibe o gráfico dos ângulos, o relatório de desvios, a validação contra o
ground-truth e o vídeo com o esqueleto sobreposto. Ideal para gravar o
vídeo-demo do Tech Challenge.

Uso:
    python -m src.video.app        # abre em http://localhost:7860
"""
from __future__ import annotations

import glob
import os
import re
import time

import gradio as gr

from .anomaly import detect_anomalies
from .cli import _downsample_video
from .keypoints import coverage, load_keypoints_dir
from .overlay import render_overlay
from .posture import compute_angles
from .report import generate_report, plot_angles
from .run_openpose import count_json, run_openpose

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT, "data", "video", "rehab24-6")
SEG_CSV = os.path.join(DATA_DIR, "Segmentation.csv")
OPENPOSE_ROOT = os.environ.get("OPENPOSE_ROOT", os.path.join(ROOT, "tools", "openpose"))
OUT_DIR = os.path.join(ROOT, "reports")


def _list_videos() -> list[str]:
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(DATA_DIR, "*.mp4")))


def process(video_file: str, frame_step: int, use_seg: bool, reuse_json: bool,
            progress=gr.Progress()):
    """Roda o pipeline e devolve (gráfico, relatório, vídeo overlay, validação)."""
    if not video_file:
        raise gr.Error("Selecione um vídeo.")
    frame_step = int(frame_step)
    video_path = os.path.join(DATA_DIR, video_file)
    video_name = os.path.splitext(video_file)[0]
    json_dir = os.path.join(OUT_DIR, "json", video_name)
    eff_fps = 30.0 / frame_step if frame_step > 1 else 30.0

    # vídeo que casa com os JSONs (subamostrado, se for o caso) + extração (OpenPose)
    t_extract = time.perf_counter()
    did_extract = False
    op_video = video_path
    if frame_step > 1:
        ds_path = os.path.join(OUT_DIR, "video", f"{video_name}-ds{frame_step}.mp4")
        if reuse_json and os.path.exists(ds_path):
            op_video = ds_path
        else:
            progress(0.0, desc="Subamostrando frames...")
            op_video = _downsample_video(video_path, os.path.join(OUT_DIR, "video"), frame_step)
            did_extract = True

    # 1. OpenPose (ou reaproveita JSONs já extraídos)
    if reuse_json and count_json(json_dir) > 0:
        progress(1.0, desc=f"Reaproveitando {count_json(json_dir)} JSONs")
    else:
        run_openpose(op_video, OPENPOSE_ROOT, json_dir, net_resolution="320x176",
                     progress_cb=lambda d, t: progress(d / t, desc=f"OpenPose {d}/{t} frames"))
        did_extract = True
    extract_seconds = (time.perf_counter() - t_extract) if did_extract else None

    # 2. Análise
    t_analysis = time.perf_counter()
    progress(0.0, desc="Calculando ângulos e desvios...")
    kp = load_keypoints_dir(json_dir)
    cov = coverage(kp)
    res = detect_anomalies(compute_angles(kp, fps=eff_fps))

    # 3. Validação + exercício (opcional)
    exercise, validation_md = None, ""
    if use_seg and os.path.exists(SEG_CSV):
        from .validate import exercise_label, load_segmentation, validate
        try:
            seg = load_segmentation(SEG_CSV, video_name.split("-Camera")[0])
            exercise = exercise_label(seg)
            _, by_class = validate(res, seg, frame_step=frame_step)
            c, i = by_class.get(1, float("nan")), by_class.get(0, float("nan"))
            sep = "✅" if (0 in by_class.index and 1 in by_class.index and i > c) else "—"
            validation_md = (f"### Validação contra o ground-truth ({exercise})\n"
                             f"Taxa média de desvio — **corretas: {c:.3f}** · "
                             f"**incorretas: {i:.3f}** {sep}")
        except Exception as e:  # noqa: BLE001 — demo: mostra o erro em vez de quebrar
            validation_md = f"_(validação indisponível: {e})_"

    # 4. Gráfico + relatório (com tempo de processamento)
    fig = os.path.join(OUT_DIR, "figures", f"angulos_{video_name}.png")
    plot_angles(res, fig, title=f"Ângulos posturais — {video_name}")
    analysis_seconds = time.perf_counter() - t_analysis
    timings: dict[str, float] = {}
    if extract_seconds is not None:
        timings["OpenPose (extração)"] = extract_seconds
    timings["Análise"] = analysis_seconds
    if extract_seconds is not None:
        timings["Total (extração + análise)"] = extract_seconds + analysis_seconds
    report_path = generate_report(res, cov, OUT_DIR, video_name, fps=eff_fps,
                                  fig_path=fig, exercise=exercise, timings=timings)
    report_md = open(report_path, encoding="utf-8").read()
    # remove a seção "## Gráfico" do texto (o gráfico já é mostrado à parte)
    report_md = re.sub(r"## Gráfico\n\n!\[[^\]]*\]\([^)]*\)\n\n", "", report_md)

    # 5. Overlay
    overlay_path = os.path.join(OUT_DIR, f"{video_name}_overlay.mp4")
    render_overlay(op_video, json_dir, overlay_path, res=res, fps=eff_fps,
                   progress_cb=lambda d, t: progress(d / t, desc=f"Overlay {d}/{t} frames"))

    return fig, validation_md, report_md, overlay_path


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Monitoramento Postural — Entrega 1") as demo:
        gr.Markdown("# 🎥 Análise Postural de Vídeo (OpenPose) — demo local\n"
                    "Selecione um vídeo do REHAB24-6, processe e veja o esqueleto, os "
                    "desvios e o relatório.")
        with gr.Row():
            video_dd = gr.Dropdown(_list_videos(), label="Vídeo (data/video/rehab24-6)")
            frame_step = gr.Slider(1, 5, value=3, step=1,
                                   label="frame-step (subamostragem p/ GPU fraca)")
        with gr.Row():
            use_seg = gr.Checkbox(value=True, label="Validar contra o ground-truth")
            reuse = gr.Checkbox(value=True, label="Reaproveitar JSONs já extraídos")
        run_btn = gr.Button("▶ Processar", variant="primary")
        with gr.Row():
            graph = gr.Image(label="Ângulos posturais", type="filepath")
            overlay = gr.Video(label="Overlay (esqueleto + desvios)")
        validation = gr.Markdown()
        report = gr.Markdown()
        run_btn.click(process, [video_dd, frame_step, use_seg, reuse],
                      [graph, validation, report, overlay])
    return demo


def main() -> None:
    build_demo().launch()


if __name__ == "__main__":
    main()
