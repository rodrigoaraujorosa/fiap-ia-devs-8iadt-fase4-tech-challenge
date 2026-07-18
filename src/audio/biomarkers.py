"""
Biomarcadores acústicos da voz (Entrega 2 — Análise de Áudio).

Extrai medidas objetivas da qualidade vocal e compara **sintomáticos** (dificuldade
respiratória ou fadiga) com **saudáveis**, usando o Coswara. É a metade da Entrega 2 que
roda local — a outra (Transcribe + Comprehend Medical) está em ``transcribe.py`` e
``comprehend.py``.

**Por que o Coswara e não as consultas.** Jitter e shimmer medem a perturbação *ciclo a
ciclo* da vibração das pregas vocais e só fazem sentido sobre **fonação sustentada** — o
paciente segurando uma vogal por alguns segundos. Em conversa espontânea, com dois
falantes, pausas e sobreposição, não há ciclos comparáveis para medir. O Coswara tem
vogais sustentadas (/a/, /e/, /o/) e respiração profunda; as consultas não têm.

**Implementação.** Jitter, shimmer e HNR vêm do **Praat** (via ``parselmouth``), que é a
implementação de referência dessas medidas. Rastreamento de F0 por frame — como o do
librosa — só permitiria uma aproximação, e o nome "jitter" implica a definição do Praat.
Os MFCC, esses sim, vêm do librosa.

Medidas extraídas:

======================  ==========================================================
``f0_mean``, ``f0_sd``  frequência fundamental: altura da voz e sua estabilidade
``jitter_local``        perturbação de **frequência** entre ciclos consecutivos
``shimmer_local``       perturbação de **amplitude** entre ciclos consecutivos
``hnr``                 razão harmônico-ruído: quanto da voz é sopro/ruído
``mfcc_1..13``          timbre (média dos coeficientes cepstrais)
``duration``            duração útil da gravação
======================  ==========================================================

Uso:
    python -m src.audio.biomarkers --root data/audio/coswara --case <id> --batch 20220224
    python -m src.audio.biomarkers --root data/audio/coswara --cohort --per-group 30
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import audio_path, build_cohort

# Faixa plausível de F0 para voz humana adulta. Limites frouxos demais fazem o rastreador
# perseguir ruído; estreitos demais cortam vozes graves ou agudas legítimas.
F0_MIN, F0_MAX = 75.0, 500.0

# Rótulos em pt-BR para os relatórios (o código fica em inglês).
MEASURE_LABELS_PT = {
    "f0_mean": "F0 média (Hz)",
    "f0_sd": "desvio de F0 (Hz)",
    "jitter_local": "jitter local",
    "shimmer_local": "shimmer local",
    "hnr": "relação harmônico-ruído (dB)",
    "duration": "duração (s)",
}


def voice_measures(wav: str | Path) -> dict:
    """
    Jitter, shimmer, HNR e F0 de uma gravação, via Praat.

    O Praat trabalha em duas etapas: primeiro estima os pontos de pulso glotal
    (``PointProcess``), depois calcula as perturbações sobre eles. Gravações sem fonação
    detectável (silêncio, só ruído) devolvem ``NaN`` em vez de zero — zero seria lido como
    "voz perfeitamente estável", o oposto da verdade.
    """
    import parselmouth
    from parselmouth.praat import call

    snd = parselmouth.Sound(str(wav))
    out: dict[str, float] = {"duration": float(snd.get_total_duration())}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pitch = snd.to_pitch(pitch_floor=F0_MIN, pitch_ceiling=F0_MAX)
        f0 = pitch.selected_array["frequency"]
        f0 = f0[f0 > 0]   # 0 marca frame sem voz (unvoiced)

        out["f0_mean"] = float(np.mean(f0)) if f0.size else float("nan")
        out["f0_sd"] = float(np.std(f0)) if f0.size else float("nan")

        try:
            pp = call(snd, "To PointProcess (periodic, cc)", F0_MIN, F0_MAX)
            # Parâmetros do Praat: janela, período mín/máx, fator máximo entre períodos.
            out["jitter_local"] = call(pp, "Get jitter (local)", 0, 0, 1e-4, 0.02, 1.3)
            out["shimmer_local"] = call([snd, pp], "Get shimmer (local)",
                                        0, 0, 1e-4, 0.02, 1.3, 1.6)
            harm = call(snd, "To Harmonicity (cc)", 0.01, F0_MIN, 0.1, 1.0)
            out["hnr"] = call(harm, "Get mean", 0, 0)
        except Exception:  # noqa: BLE001 — gravação sem fonação detectável
            out["jitter_local"] = out["shimmer_local"] = out["hnr"] = float("nan")

    return out


def spectral_measures(wav: str | Path, n_mfcc: int = 13) -> dict:
    """Média dos MFCC — descrevem o timbre, complementando as medidas de perturbação."""
    import librosa

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, sr = librosa.load(str(wav), sr=None, mono=True)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return {f"mfcc_{i + 1}": float(v) for i, v in enumerate(mfcc.mean(axis=1))}


def extract_features(wav: str | Path, with_mfcc: bool = True) -> dict:
    """Todas as medidas de uma gravação."""
    feats = voice_measures(wav)
    if with_mfcc:
        feats.update(spectral_measures(wav))
    return feats


def process_cohort(
    root: str | Path,
    batches: list[str] | None = None,
    sound: str = "vowel-a",
    per_group: int | None = None,
    with_mfcc: bool = True,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Extrai os biomarcadores de cada participante da coorte.

    O padrão é ``vowel-a``: a vogal sustentada é o substrato clássico para jitter e
    shimmer. Gravações ausentes ou ilegíveis são puladas com aviso — não interrompem o
    lote, mas também não entram silenciosamente como zero.
    """
    cohort = build_cohort(root, batches=batches, sound=sound, per_group=per_group)
    rows, skipped = [], 0

    for i, r in enumerate(cohort.itertuples(), 1):
        wav = audio_path(root, r.batch, r.id, sound)
        if not wav.exists():
            skipped += 1
            continue
        try:
            feats = extract_features(wav, with_mfcc=with_mfcc)
        except Exception as e:  # noqa: BLE001 — arquivo corrompido: registra e segue
            print(f"  [aviso] {r.id}: {type(e).__name__} — pulado")
            skipped += 1
            continue
        rows.append({"id": r.id, "group": r.group, "batch": r.batch,
                     "age": r.a, "gender": r.g, **feats})
        if progress and i % 10 == 0:
            print(f"  {i}/{len(cohort)} processados", end="\r")

    if progress:
        print(f"  {len(rows)} gravações processadas, {skipped} puladas")
    return pd.DataFrame(rows)


def compare_groups(df: pd.DataFrame, measures: list[str] | None = None) -> pd.DataFrame:
    """
    Compara sintomáticos e saudáveis em cada medida.

    Usa **Mann-Whitney U**, teste não-paramétrico: jitter e shimmer não são normalmente
    distribuídos (são razões limitadas em zero e com cauda longa), então o teste t seria
    inadequado. Devolve também o *rank-biserial*, que é o tamanho do efeito — o valor-p
    diz se há diferença, não se ela é grande.
    """
    from scipy.stats import mannwhitneyu

    if measures is None:
        measures = [m for m in MEASURE_LABELS_PT if m in df.columns and m != "duration"]

    sym = df[df["group"] == "symptomatic"]
    hea = df[df["group"] == "healthy"]

    rows = []
    for m in measures:
        a = sym[m].dropna()
        b = hea[m].dropna()
        if len(a) < 3 or len(b) < 3:
            continue
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        # rank-biserial: 0 = sem efeito, ±1 = separação total entre os grupos
        effect = 2 * u / (len(a) * len(b)) - 1
        rows.append({
            "measure": m,
            "label_pt": MEASURE_LABELS_PT.get(m, m),
            "symptomatic_median": round(float(a.median()), 4),
            "healthy_median": round(float(b.median()), 4),
            "n_symptomatic": len(a),
            "n_healthy": len(b),
            "p_value": round(float(p), 4),
            "effect_size": round(float(effect), 4),
        })
    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Biomarcadores acústicos da voz (Coswara).")
    ap.add_argument("--root", default="data/audio/coswara", help="raiz dos dados do Coswara")
    ap.add_argument("--case", help="id de um participante, para inspeção")
    ap.add_argument("--batch", help="lote do participante (com --case)")
    ap.add_argument("--sound", default="vowel-a", help="gravação a analisar")
    ap.add_argument("--cohort", action="store_true", help="processa a coorte inteira")
    ap.add_argument("--batches", nargs="*", default=["20220224", "20210406"],
                    help="lotes a considerar")
    ap.add_argument("--per-group", type=int, help="equilibra as classes com N por grupo")
    ap.add_argument("--no-mfcc", action="store_true", help="pula os MFCC (mais rápido)")
    ap.add_argument("--out", help="salva as medidas em CSV")
    args = ap.parse_args()

    if args.case:
        if not args.batch:
            ap.error("--batch é obrigatório com --case")
        wav = audio_path(args.root, args.batch, args.case, args.sound)
        if not wav.exists():
            print(f"gravação não encontrada: {wav}")
            return
        feats = extract_features(wav, with_mfcc=not args.no_mfcc)
        print(f"{args.case} / {args.sound}")
        for k, v in feats.items():
            if k.startswith("mfcc_"):
                continue
            print(f"  {MEASURE_LABELS_PT.get(k, k):32s} {v:.4f}")
        return

    if args.cohort:
        print(f"extraindo biomarcadores de '{args.sound}'...")
        df = process_cohort(args.root, batches=args.batches, sound=args.sound,
                            per_group=args.per_group, with_mfcc=not args.no_mfcc)
        if df.empty:
            print("nenhuma gravação processada")
            return

        print(f"\ncoorte: {len(df)} participantes")
        print(df["group"].value_counts().to_string())

        comp = compare_groups(df)
        print("\nsintomáticos vs. saudáveis (Mann-Whitney U):")
        print(comp.to_string(index=False))

        if args.out:
            df.to_csv(args.out, index=False)
            comp.to_csv(args.out.replace(".csv", "_comparison.csv"), index=False)
            print(f"\nsalvo em {args.out} e {args.out.replace('.csv', '_comparison.csv')}")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
