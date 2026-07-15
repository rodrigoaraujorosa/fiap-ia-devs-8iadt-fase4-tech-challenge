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
import time

from .anomaly import detect_anomalies
from .keypoints import coverage, load_keypoints_dir
from .posture import compute_angles
from .report import fmt_dur, generate_report, plot_angles


def _tqdm_progress(desc: str):
    """Callback (feito, total) -> barra tqdm criada preguiçosamente no 1º update."""
    state: dict = {"bar": None}

    def cb(done: int, total: int) -> None:
        if state["bar"] is None:
            from tqdm import tqdm
            state["bar"] = tqdm(total=total, desc=desc, unit="frame")
        bar = state["bar"]
        bar.n = min(done, total)
        bar.refresh()
        if done >= total:
            bar.close()

    return cb


def run_pipeline(json_dir: str, video_name: str, out_dir: str, fps: float = 30.0,
                 extract_seconds: float | None = None, exercise: str | None = None):
    """
    Executa parser -> ângulos -> detecção -> relatório sobre uma pasta de JSONs.
    Retorna o DataFrame ``res`` (para reuso, ex.: overlay). ``extract_seconds``
    (tempo do OpenPose) e ``exercise`` (rótulo do exercício, do Segmentation.csv)
    são opcionais e aparecem no relatório.
    """
    t0 = time.perf_counter()
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
    analysis_seconds = time.perf_counter() - t0

    timings: dict[str, float] = {}
    if extract_seconds is not None:
        timings["OpenPose (extração)"] = extract_seconds
    timings["Análise"] = analysis_seconds
    if extract_seconds is not None:
        timings["Total (extração + análise)"] = extract_seconds + analysis_seconds

    report_path = generate_report(res, cov, out_dir, video_name, fps=fps,
                                  fig_path=fig_path, timings=timings, exercise=exercise)
    print(f"\nRelatório: {report_path}\nGráfico:   {fig_path}")
    print("Tempo:")
    for etapa, seg in timings.items():
        print(f"  {etapa}: {fmt_dur(seg)}")
    return res


def _downsample_video(src: str, out_dir: str, step: int) -> str:
    """
    Escreve um vídeo com 1 a cada ``step`` frames (subamostragem para acelerar o
    OpenPose em GPU fraca). Retorna o caminho do vídeo reduzido.
    """
    import cv2

    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo: {src}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    name = os.path.splitext(os.path.basename(src))[0]
    dst = os.path.join(out_dir, f"{name}-ds{step}.mp4")
    writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), src_fps / step, (w, h))
    i = kept = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step == 0:
            writer.write(fr)
            kept += 1
        i += 1
    cap.release()
    writer.release()
    print(f"      {kept} frames mantidos (1 a cada {step}) -> {dst}")
    return dst


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
    ap.add_argument("--frame-step", type=int, default=1,
                    help="processa 1 a cada N frames (subamostragem p/ GPU fraca "
                         "com --video). Também mapeia os frames na validação. "
                         "Ex.: 3 reduz o tempo do OpenPose em ~3x")
    ap.add_argument("--segmentation",
                    help="CSV de rótulos do REHAB24-6; ativa a validação contra o "
                         "ground-truth (correto/incorreto por repetição)")
    ap.add_argument("--video-id",
                    help="id do vídeo no Segmentation.csv (padrão: derivado do nome, "
                         "ex.: PM_006-Camera17-30fps -> PM_006)")
    args = ap.parse_args()

    t_main = time.perf_counter()
    extract_seconds = None
    if args.video:
        if not args.openpose_root:
            ap.error("--openpose-root é obrigatório quando se usa --video")
        from .run_openpose import run_openpose
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        video_path = args.video
        eff_fps = args.fps
        t_extract = time.perf_counter()
        if args.frame_step > 1:
            print(f"[0/4] Subamostrando 1 a cada {args.frame_step} frames ...")
            video_path = _downsample_video(args.video, os.path.join(args.out, "video"),
                                           args.frame_step)
            eff_fps = args.fps / args.frame_step
        json_dir = os.path.join(args.out, "json", video_name)
        run_openpose(video_path, args.openpose_root, json_dir,
                     net_resolution=args.net_resolution,
                     progress_cb=_tqdm_progress("OpenPose"))
        extract_seconds = time.perf_counter() - t_extract
    else:
        if args.overlay:
            ap.error("--overlay requer --video (precisamos do vídeo original)")
        json_dir = args.json_dir
        video_name = os.path.basename(os.path.normpath(json_dir)).replace("_json", "")
        # em --json-dir os JSONs já existem; frame_step serve só p/ mapear a validação
        video_path, eff_fps = None, args.fps

    # exercício (rótulo do dataset) para o relatório, quando houver segmentação
    exercise = None
    if args.segmentation:
        from .validate import exercise_label, load_segmentation
        vid = args.video_id or video_name.split("-Camera")[0]
        try:
            exercise = exercise_label(load_segmentation(args.segmentation, vid))
        except (ValueError, KeyError, FileNotFoundError):
            exercise = None

    res = run_pipeline(json_dir, video_name, args.out, fps=eff_fps,
                       extract_seconds=extract_seconds, exercise=exercise)

    extra_seconds = 0.0
    if args.overlay:
        from .overlay import render_overlay
        overlay_path = os.path.join(args.out, f"{video_name}_overlay.mp4")
        t = time.perf_counter()
        render_overlay(video_path, json_dir, overlay_path, res=res, fps=eff_fps,
                       progress_cb=_tqdm_progress("Overlay"))
        extra_seconds += time.perf_counter() - t

    if args.segmentation:
        t = time.perf_counter()
        _run_validation(res, args.segmentation, video_name, args.video_id,
                        args.frame_step, args.out)
        extra_seconds += time.perf_counter() - t

    print(f"\nTempo total (fim-a-fim): {fmt_dur(time.perf_counter() - t_main)}")
    if extra_seconds > 1:
        print(f"  (o 'Total' do relatório cobre extração + análise; "
              f"overlay + validação somam +{fmt_dur(extra_seconds)})")


def _run_validation(res, seg_csv: str, video_name: str, video_id: str | None,
                    frame_step: int, out_dir: str) -> None:
    """Valida os desvios detectados contra os rótulos correto/incorreto do REHAB24-6."""
    from .validate import load_segmentation, validate

    vid = video_id or video_name.split("-Camera")[0]
    print(f"\n[validação] ground-truth: video_id={vid}, frame_step={frame_step}")
    seg = load_segmentation(seg_csv, vid)
    per_rep, by_class = validate(res, seg, frame_step=frame_step)

    out_csv = os.path.join(out_dir, f"validacao_{video_name}.csv")
    per_rep.to_csv(out_csv, index=False)
    print("      taxa média de frames com desvio por classe:")
    for cls, label in ((1, "corretas  "), (0, "incorretas")):
        if cls in by_class.index:
            print(f"        {label}: {by_class[cls]:.3f}")
    if {0, 1} <= set(by_class.index):
        sep = "OK (incorretas > corretas)" if by_class[0] > by_class[1] else "sem separação"
        print(f"      separação: {sep}")
    print(f"      detalhe por repetição salvo em {out_csv}")


if __name__ == "__main__":
    main()
