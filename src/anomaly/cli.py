"""
Pipeline fim-a-fim da Entrega 3 (Detecção de Anomalias).

Roda as três subtarefas e gera o relatório único para a equipe médica:

    1. Movimentação   UCI HAR          -> IsolationForest treinado só em repouso
    2. Sinais vitais  Challenge 2019   -> IsolationForest sobre desvios por paciente
    3. Prescrições    Challenge 2019   -> degrau de dose (FiO2), variável derivada

Este é o **único ponto de entrada** da entrega: ``vitals.py``, ``movement.py``,
``prescriptions.py`` e ``report.py`` são módulos de biblioteca, sem CLI própria — mesma
organização das Entregas 1 e 2.

O detector de sinais vitais é **treinado e salvo**: a coorte de treino tem apenas
pacientes que nunca desenvolveram sepse, e a avaliação ocorre sobre pacientes retidos,
que o modelo não viu. Depois de treinado, o modo de monitoramento pontua **um paciente
por vez**, que é como o sistema operaria em leito.

Movimentação e sinais vitais vêm de datasets distintos, sem correspondência de
indivíduos — não existe o paciente ``p000009`` no UCI HAR. Por isso o monitoramento tem
dois comandos: ``--monitor`` cobre as duas subtarefas do Challenge 2019 (vitais e dose
prescrita, que descrevem o mesmo paciente) e ``--monitor-subject`` cobre a movimentação.

Uso:
    python -m src.anomaly.cli --train                  # treina e salva os detectores
    python -m src.anomaly.cli --monitor p000009        # vitais + prescrições de 1 paciente
    python -m src.anomaly.cli --monitor-subject 2      # movimentação de 1 sujeito
    python -m src.anomaly.cli                          # avaliação completa + relatório
    python -m src.anomaly.cli --only movement          # uma subtarefa
    python -m src.anomaly.cli --limit 5000             # tamanho da coorte

Não custa dinheiro: roda inteiramente local.
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from . import alerts, movement, prescriptions, vitals
from .report import (FIGURES_DIR, REPORT_PATH, build_report, plot_monitor,
                     write_report)

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
    print(f"      treino: {m['train_patients']} pacientes SEM sepse "
          f"({m['train_hours']} horas, {m['features_used']} features)")
    print(f"      teste : {m['patients']} pacientes retidos | {m['rows']} horas | "
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
    print(f"      {m['escalations']} escalonamentos | {m['reductions']} reduções")
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


def train_and_save(vitals_dir: str, har_dir: str, limit: int | None,
                   contamination: float) -> None:
    """Treina os dois detectores no respectivo padrão de normalidade e persiste."""
    t0 = time.perf_counter()

    print("Detector de sinais vitais")
    raw = vitals.load_dataset(vitals_dir, limit=limit)
    treino, teste = vitals.split_cohorts(vitals.prepare(raw))
    det_v = vitals.fit(treino, contamination=contamination)
    caminho_v = vitals.save(det_v)
    print(f"  coorte de treino : {det_v.trained_on['patients']} pacientes sem sepse, "
          f"{det_v.trained_on['hours']} horas")
    print(f"  retidos p/ teste : {teste['patient'].nunique()} pacientes")
    print(f"  features usadas  : {len(det_v.features)}")
    print(f"  limiar de alerta : {det_v.threshold:.4f}")
    print(f"  salvo em         : {caminho_v}")

    print("\nDetector de movimentação")
    X_train, y_train, subj_train = movement.load_split(har_dir, "train")
    det_m = movement.fit(X_train, y_train, contamination=contamination)
    caminho_m = movement.save(det_m)
    _, _, subj_test = movement.load_split(har_dir, "test")
    print(f"  coorte de treino : {det_m.trained_on['samples']} amostras de repouso, "
          f"{subj_train['subject'].nunique()} sujeitos")
    print(f"  retidos p/ teste : {subj_test['subject'].nunique()} sujeitos — "
          f"{', '.join(str(s) for s in sorted(subj_test['subject'].unique()))}")
    print(f"  salvo em         : {caminho_m}")

    print("\nA subtarefa de prescrições não tem modelo a treinar: a detecção é uma regra")
    print("de degrau na dose, idêntica para um paciente ou para a coorte inteira.")
    print(f"\n({_fmt(time.perf_counter() - t0)})")


def monitor(patient_id: str, data_dir: str) -> None:
    """
    Monitoramento de um paciente de UTI com os modelos já treinados.

    Cobre as duas subtarefas que vêm do Challenge 2019 — sinais vitais e evolução da
    dose prescrita — porque ambas descrevem o mesmo paciente. A movimentação usa outro
    dataset, sem correspondência de indivíduos, e tem comando próprio
    (``--monitor-subject``).
    """
    r = vitals.monitor_patient(patient_id, data_dir)
    s = r["summary"]
    serie = r["data"]

    print(f"\n{'=' * 66}")
    print(f"Paciente {s['patient']} — {s['hours']} horas de internação")
    print(f"{'=' * 66}")

    # ---------------------------------------------------------------- vitais
    print(f"\nSINAIS VITAIS   modelo: {vitals.MODEL_PATH}")
    if not s["alerts"]:
        print("  Nenhum alerta — série dentro do padrão aprendido.")
    else:
        horas = s["alert_hours"]
        print(f"  {s['alerts']} horas em alerta ({s['alert_rate']:.1%} da internação)")
        print(f"    horas {', '.join(str(h) for h in horas[:20])}"
              + (" ..." if len(horas) > 20 else ""))

    # ----------------------------------------------------------- prescrições
    px_r = prescriptions.monitor_patient(serie)
    px = px_r["summary"]
    print("\nPRESCRIÇÕES     dose de oxigênio (FiO2), regra de degrau")
    if not px["monitored"]:
        print(f"  Fora de monitoramento — {px['observations']} registros de dose "
              f"(mínimo {prescriptions.MIN_OBSERVATIONS}).")
    else:
        print(f"  {px['observations']} registros, dose entre {px['dose_min']:.2f} "
              f"e {px['dose_max']:.2f}")
        if px["escalations"]:
            print(f"  {px['escalations']} escalonamento(s) nas horas "
                  f"{', '.join(str(h) for h in px['escalation_hours'])}")
        else:
            print("  Nenhum escalonamento brusco.")
        if px["reductions"]:
            print(f"  {px['reductions']} redução(ões) de dose — não geram alerta.")

    # ------------------------------------------------------------ conferência
    print("\nCONFERÊNCIA COM O GROUND-TRUTH (não usado na detecção)")
    if s["developed_sepsis"]:
        print(f"  Sepse registrada a partir da hora {s['onset_hour']}.")
        if s.get("warned"):
            print(f"  Vitais    — alerta {s['lead_hours']} h antes do início.")
        else:
            print(f"  Vitais    — sem alerta na janela de "
                  f"{vitals.LEAD_WINDOW_HOURS} h que antecede o início.")
        if px.get("lead_hours"):
            print(f"  Prescrição — escalonamento {px['lead_hours']} h antes do início.")
        elif px["escalations"]:
            print("  Prescrição — escalonamentos fora da janela de 48 h.")
    else:
        print("  Paciente não desenvolveu sepse.")

    # a figura usa a série já pontuada e com a dose anotada, para que os dois painéis
    # compartilhem exatamente os mesmos instantes
    fig = plot_monitor(px_r["data"], patient_id,
                       str(FIGURES_DIR / f"monitor_{patient_id}.png"))
    if fig:
        print(f"\n  Figura: {fig}")
    print()


def painel(data_dir: str, limit: int | None) -> None:
    """Fila de plantão da coorte retida, ordenada por prioridade."""
    t0 = time.perf_counter()
    fila = alerts.scan(data_dir, limit=limit)
    print()
    print(alerts.render(fila))
    print(f"({_fmt(time.perf_counter() - t0)})")


def monitor_subject(subject_id: int, har_dir: str) -> None:
    """Monitoramento da movimentação de um sujeito do conjunto de teste."""
    r = movement.monitor_subject(subject_id, har_dir)
    s = r["summary"]
    res = r["data"]

    print(f"\n{'=' * 66}")
    print(f"Sujeito {s['subject']} — {s['windows']} janelas de leitura")
    print(f"{'=' * 66}")
    print(f"\nMOVIMENTAÇÃO    modelo: {movement.MODEL_PATH}")
    print(f"  {s['alerts']} janelas em alerta ({s['alert_rate']:.1%} das leituras)")

    print("\nCONFERÊNCIA COM O GROUND-TRUTH (não usado na detecção)")
    tabela = (res.groupby("activity")
                 .agg(janelas=("is_anomaly", "size"), alerta=("is_anomaly", "mean")))
    for atividade, linha in tabela.sort_values("alerta", ascending=False).iterrows():
        marca = "movimento" if atividade not in movement.REST_ACTIVITIES else "repouso  "
        print(f"  {marca}  {atividade:20s} {int(linha['janelas']):4d} janelas  "
              f"{linha['alerta']:6.1%} em alerta")
    if s["recall_movement"] is not None:
        print(f"\n  Recall sobre marcha: {s['recall_movement']:.1%} | "
              f"falso alarme no repouso: {s['false_alarm_rate']:.1%}")
    print()


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
    ap.add_argument("--train", action="store_true",
                    help="treina o detector de vitais e salva em models/")
    ap.add_argument("--monitor", metavar="PACIENTE",
                    help="alerta de um paciente de UTI: sinais vitais + prescrições")
    ap.add_argument("--monitor-subject", metavar="ID", type=int,
                    help="alerta de movimentação de um sujeito do UCI HAR")
    ap.add_argument("--alerts", action="store_true",
                    help="painel de plantão: fila de alertas da coorte, por prioridade")
    args = ap.parse_args()

    if args.train:
        train_and_save(args.vitals_data, args.har_data, args.limit, args.contamination)
        return

    if args.monitor:
        monitor(args.monitor, args.vitals_data)
        return

    if args.monitor_subject is not None:
        monitor_subject(args.monitor_subject, args.har_data)
        return

    if args.alerts:
        painel(args.vitals_data, args.limit)
        return

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
