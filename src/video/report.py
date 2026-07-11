"""
Geração de relatório automático de desvios posturais.

Produz dois artefatos a partir do DataFrame anotado por :mod:`anomaly`:
  - um **gráfico** (PNG) com os ângulos ao longo do tempo e os frames anômalos
    destacados;
  - um **relatório em Markdown** resumindo cobertura, estatísticas dos ângulos e
    os principais eventos de desvio (intervalos contíguos de anomalia).
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # backend headless (roda em servidor/Colab sem display)
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .posture import angle_columns  # noqa: E402

# Ângulos mais informativos para o gráfico dos exercícios do REHAB24-6 (ex.: agachamento).
_PLOT_ANGLES = ["trunk_inclination", "r_hip", "l_hip", "r_knee", "l_knee"]


def _contiguous_runs(mask: pd.Series) -> list[tuple[int, int]]:
    """Converte uma máscara booleana por frame em intervalos (início, fim)."""
    runs: list[tuple[int, int]] = []
    start = None
    for frame, flag in mask.items():
        if flag and start is None:
            start = frame
        elif not flag and start is not None:
            runs.append((start, prev))
            start = None
        prev = frame
    if start is not None:
        runs.append((start, prev))
    return runs


def plot_angles(res: pd.DataFrame, out_path: str, title: str = "") -> str:
    """Salva o gráfico de ângulos com os frames anômalos marcados. Retorna o path."""
    cols = [c for c in _PLOT_ANGLES if c in res.columns]
    fig, ax = plt.subplots(figsize=(12, 5))
    for col in cols:
        ax.plot(res["time_s"], res[col], label=col, alpha=0.85, linewidth=1.2)

    anom = res[res["is_anomaly"] == 1]
    if not anom.empty:
        ax.scatter(anom["time_s"], [ax.get_ylim()[1]] * len(anom),
                   marker="v", color="red", s=25, label="anomalia", zorder=5)

    ax.set_xlabel("tempo (s)")
    ax.set_ylabel("ângulo (graus)")
    ax.set_title(title or "Ângulos posturais ao longo do tempo")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def generate_report(
    res: pd.DataFrame,
    coverage: pd.Series,
    out_dir: str,
    video_name: str,
    fps: float = 30.0,
    fig_path: str | None = None,
) -> str:
    """
    Escreve o relatório Markdown em ``out_dir`` e retorna o caminho do arquivo.

    ``res`` deve conter as colunas produzidas por :func:`anomaly.detect_anomalies`;
    ``coverage`` é a saída de :func:`keypoints.coverage`.
    """
    os.makedirs(out_dir, exist_ok=True)
    feat_cols = [c for c in angle_columns(res)
                 if c not in ("z_anomaly", "iso_anomaly", "is_anomaly",
                              "worst_angle", "anomaly_score")]

    n_frames = len(res)
    n_anom = int(res["is_anomaly"].sum())
    runs = _contiguous_runs(res["is_anomaly"] == 1)

    lines: list[str] = []
    lines.append(f"# Relatório automático de desvios posturais — {video_name}\n")
    lines.append(f"- **Frames analisados:** {n_frames} (~{n_frames / fps:.1f} s a {fps:.0f} fps)")
    lines.append(f"- **Frames sinalizados como anomalia:** {n_anom} "
                 f"({n_anom / n_frames:.1%})")
    lines.append(f"- **Eventos de desvio (intervalos contíguos):** {len(runs)}\n")

    lines.append("## Cobertura de detecção das juntas (%)\n")
    low = coverage[coverage < 60]
    if not low.empty:
        lines.append("⚠️ Juntas com baixa detecção (<60% dos frames) — podem "
                     "prejudicar os ângulos associados:\n")
        for name, val in low.items():
            lines.append(f"- {name}: {val}%")
    else:
        lines.append("Todas as juntas detectadas em ≥60% dos frames. ✅")
    lines.append("")

    lines.append("## Estatística dos ângulos (graus)\n")
    stats = res[feat_cols].describe().loc[["mean", "min", "max"]].round(1).T
    stats["amplitude"] = (stats["max"] - stats["min"]).round(1)
    lines.append(stats.to_markdown())
    lines.append("")

    lines.append("## Principais eventos de desvio\n")
    if runs:
        lines.append("| início (s) | fim (s) | duração (s) | ângulo predominante | |z| máx |")
        lines.append("|---|---|---|---|---|")
        events = []
        for start, end in runs:
            seg = res.loc[start:end]
            worst = seg["worst_angle"].mode()
            worst = worst.iloc[0] if not worst.empty else "-"
            events.append((start, end, seg["anomaly_score"].max()))
            lines.append(f"| {start / fps:.1f} | {end / fps:.1f} | "
                         f"{(end - start + 1) / fps:.1f} | {worst} | "
                         f"{seg['anomaly_score'].max():.1f} |")
    else:
        lines.append("Nenhum desvio significativo detectado.")
    lines.append("")

    if fig_path:
        # barras normais para o link funcionar em qualquer renderizador (GitHub etc.)
        rel = os.path.relpath(fig_path, out_dir).replace(os.sep, "/")
        lines.append("## Gráfico\n")
        lines.append(f"![ângulos]({rel})")

    out_path = os.path.join(out_dir, f"relatorio_{video_name}.md")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return out_path
