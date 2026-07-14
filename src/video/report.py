"""
Geração de relatório automático de desvios posturais.

Produz dois artefatos a partir do DataFrame anotado por :mod:`anomaly`:
  - um **gráfico** (PNG) com os ângulos ao longo do tempo e os frames anômalos
    destacados;
  - um **relatório em Markdown** para a equipe médica, na ordem: gráfico → análise
    interpretativa (gerada automaticamente) → cobertura de detecção → estatística dos
    ângulos → resumo e principais eventos de desvio.
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

# Nomes em português das articulações/ângulos (para a equipe médica).
ANGLE_PT = {
    "r_elbow": "Cotovelo D", "l_elbow": "Cotovelo E",
    "r_shoulder": "Ombro D", "l_shoulder": "Ombro E",
    "r_hip": "Quadril D", "l_hip": "Quadril E",
    "r_knee": "Joelho D", "l_knee": "Joelho E",
    "trunk_inclination": "Inclinação do tronco",
}


def fmt_dur(seconds: float) -> str:
    """Formata uma duração em segundos de forma legível (mostra min se >= 60s)."""
    if seconds >= 60:
        return f"{seconds:.0f} s (~{seconds / 60:.1f} min)"
    return f"{seconds:.1f} s"


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
    """Salva o gráfico de ângulos com os instantes de desvio destacados. Retorna o path."""
    cols = [c for c in _PLOT_ANGLES if c in res.columns]
    # garante que a articulação mais sinalizada (citada na Análise) apareça no gráfico,
    # evitando que a análise mencione um ângulo que não está plotado
    anoms = res[res["is_anomaly"] == 1]
    if not anoms.empty:
        top = anoms["worst_angle"].value_counts().index[0]
        if top in res.columns and top not in cols:
            cols.append(top)

    fig, ax = plt.subplots(figsize=(13, 5))
    for col in cols:
        ax.plot(res["time_s"], res[col], label=ANGLE_PT.get(col, col),
                alpha=0.85, linewidth=1.2)

    # sombreia as janelas de desvio (intervalos contíguos) — facilita localizá-las
    runs = _contiguous_runs(res["is_anomaly"] == 1)
    for i, (start, end) in enumerate(runs):
        ax.axvspan(res.loc[start, "time_s"], res.loc[end, "time_s"],
                   color="red", alpha=0.10, lw=0,
                   label="instantes de desvio" if i == 0 else None)

    # marcador no topo para cada instante sinalizado
    anom = res[res["is_anomaly"] == 1]
    if not anom.empty:
        ax.scatter(anom["time_s"], [ax.get_ylim()[1]] * len(anom),
                   marker="v", color="red", s=16, zorder=5)

    ax.set_xlabel("tempo (s)")
    ax.set_ylabel("ângulo (graus)")
    ax.set_title(title or "Ângulos posturais ao longo do tempo")
    # legenda FORA da área de plotagem (à direita), sem sobrepor as linhas
    ax.legend(loc="upper left", bbox_to_anchor=(1.005, 1.0), fontsize=8,
              framealpha=0.9)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")  # inclui a legenda externa
    plt.close(fig)
    return out_path


def _build_analysis(res: pd.DataFrame, feat_cols: list[str]) -> list[str]:
    """
    Gera a análise interpretativa (texto) a partir dos dados detectados.

    Escrita para a equipe médica: destaca a articulação mais afetada, onde os
    desvios se concentram no tempo, o pico de severidade e a inclinação do tronco.
    """
    n = len(res)
    anoms = res[res["is_anomaly"] == 1]
    n_anom = len(anoms)
    dur = float(res["time_s"].max()) if n else 0.0

    if n_anom == 0:
        return ["Não foram detectados desvios posturais significativos em relação ao padrão "
                "predominante do vídeo. A execução manteve-se dentro da faixa esperada para "
                "os ângulos monitorados."]

    lines = [
        f"Ao longo dos {dur:.0f}s analisados, o sistema sinalizou {n_anom} de {n} instantes "
        f"({n_anom / n:.1%}) como **desvio postural** em relação ao padrão predominante do "
        f"próprio vídeo. Os pontos abaixo resumem os achados para orientar a revisão clínica.",
        "",
    ]

    # Articulação mais afetada (moda do ângulo de maior desvio entre os frames anômalos).
    vc = anoms["worst_angle"].value_counts()
    top_ang = vc.index[0]
    lines.append(f"- **Articulação mais afetada:** {ANGLE_PT.get(top_ang, top_ang)} — "
                 f"predominante em {int(vc.iloc[0])} dos {n_anom} instantes sinalizados "
                 f"({vc.iloc[0] / n_anom:.0%}).")

    # Concentração temporal: janela de tempo com maior acúmulo de desvios.
    bin_s = max(5.0, dur / 12) if dur > 0 else 5.0
    tb = (res["time_s"] // bin_s).astype(int)
    by_bin = res["is_anomaly"].groupby(tb).sum()
    top_bin = int(by_bin.idxmax())
    w0, w1 = top_bin * bin_s, min((top_bin + 1) * bin_s, dur)
    win_rate = res.loc[tb == top_bin, "is_anomaly"].mean()
    lines.append(f"- **Concentração temporal:** o maior acúmulo de desvios ocorre entre "
                 f"{w0:.0f}s e {w1:.0f}s ({win_rate:.0%} dos instantes dessa janela sinalizados).")

    # Pico de severidade.
    idx = res["anomaly_score"].idxmax()
    peak_ang = res.loc[idx, "worst_angle"]
    lines.append(f"- **Pico de severidade:** em t={res.loc[idx, 'time_s']:.0f}s, no(a) "
                 f"{ANGLE_PT.get(peak_ang, peak_ang)} (z={res.loc[idx, 'anomaly_score']:.1f}).")

    # Inclinação do tronco (indicador de projeção para a frente).
    if "trunk_inclination" in feat_cols:
        mt = float(res["trunk_inclination"].max())
        flag = " — inclinação acentuada, possível projeção do tronco à frente" if mt >= 30 else ""
        lines.append(f"- **Inclinação máxima do tronco:** {mt:.0f}°{flag}.")

    # Maior amplitude de movimento.
    amp = res[feat_cols].max() - res[feat_cols].min()
    amax = amp.idxmax()
    lines.append(f"- **Maior amplitude de movimento:** {ANGLE_PT.get(amax, amax)} "
                 f"({amp[amax]:.0f}°).")

    lines.append("")
    lines.append("Recomenda-se que a equipe revise a execução nos períodos destacados em "
                 "vermelho no gráfico. **Este relatório é gerado automaticamente e não "
                 "substitui a avaliação de um profissional de saúde.**")
    return lines


def generate_report(
    res: pd.DataFrame,
    coverage: pd.Series,
    out_dir: str,
    video_name: str,
    fps: float = 30.0,
    fig_path: str | None = None,
    timings: dict[str, float] | None = None,
    exercise: str | None = None,
) -> str:
    """
    Escreve o relatório Markdown em ``out_dir`` e retorna o caminho do arquivo.

    Ordem das seções: Gráfico → Análise → Cobertura → Estatística → Resumo e eventos.
    ``res`` deve conter as colunas produzidas por :func:`anomaly.detect_anomalies`;
    ``coverage`` é a saída de :func:`keypoints.coverage`. ``timings`` (opcional) é um
    dict {etapa: segundos} exibido no resumo como tempo de processamento.
    """
    os.makedirs(out_dir, exist_ok=True)
    feat_cols = [c for c in angle_columns(res)
                 if c not in ("z_anomaly", "iso_anomaly", "is_anomaly",
                              "worst_angle", "anomaly_score")]

    n_frames = len(res)
    n_anom = int(res["is_anomaly"].sum())
    runs = _contiguous_runs(res["is_anomaly"] == 1)

    lines: list[str] = [f"# Relatório automático de desvios posturais — {video_name}\n"]
    if exercise:
        lines.append(f"**Exercício:** {exercise} — *rótulo do dataset, não detectado "
                     f"automaticamente*\n")

    # 1. Gráfico (primeiro).
    if fig_path:
        # barras normais para o link funcionar em qualquer renderizador (GitHub etc.)
        rel = os.path.relpath(fig_path, out_dir).replace(os.sep, "/")
        lines.append("## Gráfico\n")
        lines.append(f"![ângulos posturais]({rel})\n")

    # 2. Análise (logo abaixo do gráfico).
    lines.append("## Análise\n")
    lines.extend(_build_analysis(res, feat_cols))
    lines.append("")

    # 3. Cobertura de detecção das juntas.
    lines.append("## Cobertura de detecção das juntas (%)\n")
    low = coverage[coverage < 60]
    if not low.empty:
        lines.append("Juntas com baixa detecção (<60% dos frames) — podem prejudicar os "
                     "ângulos associados:\n")
        for name, val in low.items():
            lines.append(f"- {name}: {val}%")
    else:
        lines.append("Todas as juntas detectadas em ≥60% dos frames.")
    lines.append("")

    # 4. Estatística dos ângulos (com nome em português e cabeçalhos traduzidos).
    lines.append("## Estatística dos ângulos (graus)\n")
    stats = res[feat_cols].agg(["mean", "min", "max"]).T
    stats["amplitude"] = stats["max"] - stats["min"]
    stats = stats.round(1)
    stats.insert(0, "Articulação", [ANGLE_PT.get(a, a) for a in stats.index])
    stats.columns = ["Articulação", "Média", "Mín", "Máx", "Amplitude"]
    stats.index.name = "Variável"
    lines.append(stats.to_markdown())
    lines.append("")

    # 5. Resumo e principais eventos de desvio (no final).
    lines.append("## Resumo e principais eventos de desvio\n")
    lines.append(f"- **Frames analisados:** {n_frames} (~{n_frames / fps:.1f} s a {fps:.0f} fps)")
    lines.append(f"- **Frames sinalizados como desvio:** {n_anom} ({n_anom / n_frames:.1%})")
    lines.append(f"- **Eventos de desvio (intervalos contíguos):** {len(runs)}")
    if timings:
        partes = " · ".join(f"{etapa} {fmt_dur(seg)}" for etapa, seg in timings.items())
        lines.append(f"- **Tempo de processamento:** {partes}")
    lines.append("")

    if runs:
        lines.append("| Início (s) | Fim (s) | Duração (s) | Articulação predominante | Severidade (z máx) |")
        lines.append("|---|---|---|---|---|")
        for start, end in runs:
            seg = res.loc[start:end]
            worst = seg["worst_angle"].mode()
            worst = worst.iloc[0] if not worst.empty else "-"
            lines.append(f"| {start / fps:.1f} | {end / fps:.1f} | "
                         f"{(end - start + 1) / fps:.1f} | {ANGLE_PT.get(worst, worst)} | "
                         f"{seg['anomaly_score'].max():.1f} |")
    else:
        lines.append("Nenhum desvio significativo detectado.")
    lines.append("")

    out_path = os.path.join(out_dir, f"relatorio_{video_name}.md")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return out_path
