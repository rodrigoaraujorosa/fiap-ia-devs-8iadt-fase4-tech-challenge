"""
Pipeline fim-a-fim da Entrega 3 (Detecção de Anomalias).

Roda as três subtarefas e gera o relatório único para a equipe médica:

    1. Movimentação   UCI HAR          -> IsolationForest treinado só em repouso
    2. Sinais vitais  Challenge 2019   -> IsolationForest sobre desvios por paciente
    3. Prescrições    Challenge 2019   -> degrau de dose (FiO2), variável derivada

Este é o **único ponto de entrada** da entrega: ``vitals.py``, ``movement.py``,
``prescriptions.py`` e ``report.py`` são módulos de biblioteca, sem CLI própria — mesma
organização das Entregas 1 e 2.

Uso:
    python -m src.anomaly.cli                          # tudo, com o relatório
    python -m src.anomaly.cli --only movement          # uma subtarefa
    python -m src.anomaly.cli --limit 1000             # mais pacientes
    python -m src.anomaly.cli --patient p000009        # figura de um paciente

Não custa dinheiro: roda inteiramente local.
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from . import movement, prescriptions, vitals
from .report import FIGURES_DIR, REPORT_PATH, build_report, write_report

DEFAULT_VITALS_DIR = "data/anomaly/challenge2019"
DEFAULT_HAR_DIR = "data/anomaly/uci_har/UCI HAR Dataset"


def _fmt(seg: float) -> str:
    return f"{seg:.1f} s" if seg < 60 else f"{seg / 60:.1f} min"


def run_movement(data_root: str, contamination: float) -> dict:
    print("\n[1/3] Movimentação — UCI HAR")
    t0 = time.perf_counter()
    r = movement.run(data_root, contamination=contamination)
    m = r["metrics"]
    print(f"      treino: {m['train_rest_samples']} amostras de repouso "
          f"({m['features']} features, {m['subjects_train']} sujeitos)")
    print(f"      precisão {m['precision']:.3f} | recall {m['recall']:.3f} | "
          f"F1 {m['f1']:.3f} | AUC {m['roc_auc']:.4f}")
    print(f"      falso alarme no repouso: {m['false_alarm_rate']:.2%}")
    print(f"      ({_fmt(time.perf_counter() - t0)})")
    return r


def run_vitals(data_dir: str, limit: int | None, contamination: float) -> dict:
    print("\n[2/3] Sinais vitais — PhysioNet Challenge 2019")
    t0 = time.perf_counter()
    r = vitals.run(data_dir, limit=limit, contamination=contamination)
    m = r["metrics"]
    print(f"      {m['patients']} pacientes | {m['rows']} horas | "
          f"{m['sepsis_patients']} com sepse")
    print(f"      AUC {m['roc_auc']:.4f} | AUPRC {m['auprc']:.4f} "
          f"(prevalência {m['prevalence']:.4f})")
    print(f"      alertados {m['sepsis_patients_alerted']}/{m['sepsis_patients']} | "
          f"avisados na janela de {m['lead_window_hours']} h "
          f"{m['sepsis_patients_warned']} | lead mediano {m['lead_median_hours']} h")
    print(f"      ({_fmt(time.perf_counter() - t0)})")
    return r


def run_prescriptions(df: pd.DataFrame) -> dict:
    print("\n[3/3] Prescrições — variável derivada (FiO2)")
    t0 = time.perf_counter()
    r = prescriptions.run(df)
    m = r["metrics"]
    print(f"      {m['eligible_patients']} pacientes elegíveis | "
          f"cobertura da FiO2 {m['dose_coverage']:.1%}")
    print(f"      {m['escalations']} escalonamentos | {m['weanings']} desmames")
    print(f"      sepse entre os que escalonaram {m['sepsis_rate_escalated']:.1%} "
          f"vs {m['sepsis_rate_not_escalated']:.1%} entre os que não")
    print(f"      ({_fmt(time.perf_counter() - t0)})")
    return r


def _pick_patient(v: dict) -> str | None:
    """
    Escolhe um paciente ilustrativo: sepse com alerta **dentro da janela**.

    Ordenar pelo maior lead escolheria justamente o caso degenerado — um alerta muito
    antes do evento, sem relação com ele. Entre os avisados a tempo, prefere o de maior
    antecedência, que é o que ilustra bem a figura.
    """
    lt = v["lead_time"]
    bons = lt[lt["alert_in_window"]] if len(lt) else lt
    if bons.empty:
        return None
    return str(bons.sort_values("lead_hours", ascending=False).iloc[0]["patient"])


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pipeline da detecção de anomalias (vitais, movimentação, prescrições).")
    ap.add_argument("--vitals-data", default=DEFAULT_VITALS_DIR,
                    help="pasta com os .psv do Challenge 2019")
    ap.add_argument("--har-data", default=DEFAULT_HAR_DIR,
                    help='pasta "UCI HAR Dataset"')
    ap.add_argument("--limit", type=int, default=300,
                    help="máximo de pacientes do Challenge 2019 (padrão: 300)")
    ap.add_argument("--contamination", type=float, default=0.05,
                    help="fração de amostras sinalizadas (padrão: 0.05)")
    ap.add_argument("--only", choices=["movement", "vitals", "prescriptions"],
                    help="roda só uma subtarefa (não gera o relatório)")
    ap.add_argument("--patient", help="paciente para a figura da série de vitais")
    ap.add_argument("--no-report", action="store_true", help="não escreve o relatório")
    args = ap.parse_args()

    t0 = time.perf_counter()
    figuras: dict[str, str] = {}

    mv = v = px = None
    comparacao = pd.DataFrame()

    if args.only in (None, "movement"):
        mv = run_movement(args.har_data, args.contamination)
        figuras["movement"] = movement.plot(
            mv["results"], str(FIGURES_DIR / "anomalia_movimentacao.png"))

    if args.only in (None, "vitals", "prescriptions"):
        raw = vitals.load_dataset(args.vitals_data, limit=args.limit)

    if args.only in (None, "vitals"):
        v = run_vitals(args.vitals_data, args.limit, args.contamination)
        alvo = args.patient or _pick_patient(v)
        if alvo:
            caminho = vitals.plot_patient(
                v["data"], alvo, str(FIGURES_DIR / "anomalia_vitais.png"))
            if caminho:
                figuras["vitals"] = caminho
                print(f"      figura: {caminho} (paciente {alvo})")
        print("\n      comparação vitais x laboratório:")
        comparacao = vitals.compare_feature_sets(
            vitals.prepare(raw), contamination=args.contamination)
        print(comparacao.to_string(index=False))

    if args.only in (None, "prescriptions"):
        px = run_prescriptions(raw)
        lt = px["lead_time"]
        if len(lt):
            alvo = str(lt.sort_values("lead_hours", ascending=False).iloc[0]["patient"])
            caminho = prescriptions.plot_patient(
                px["data"], alvo, str(FIGURES_DIR / "anomalia_prescricoes.png"))
            if caminho:
                figuras["prescriptions"] = caminho
                print(f"      figura: {caminho} (paciente {alvo})")

    if args.only is None and not args.no_report:
        md = build_report(v, mv, px, comparacao, figuras)
        caminho = write_report(md)
        print(f"\nRelatório em {caminho}")

    print(f"\nTotal: {_fmt(time.perf_counter() - t0)}")


if __name__ == "__main__":
    main()
