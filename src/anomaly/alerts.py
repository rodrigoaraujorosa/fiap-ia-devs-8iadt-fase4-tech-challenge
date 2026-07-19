"""
Geração de alertas automáticos para a equipe médica (Entrega 3).

O enunciado pede, no item 3, "gerar alertas automáticos para a equipe médica com base nas
anomalias detectadas". Este módulo é essa camada: recebe as anomalias já detectadas pelos
outros módulos e as transforma em uma fila de plantão, ordenada por prioridade.

**A prioridade não é arbitrária — sai da confiabilidade medida de cada modalidade** (§5.7
do relatório técnico). Tratar um detector com AUC 0,555 do mesmo modo que um com AUC
0,9999 produziria uma fila em que o ruído dos sinais vitais soterra os achados confiáveis
de movimentação:

- **Movimentação** (AUC 0,9999) — separação praticamente completa; alerta automático.
- **Prescrições** — sinal fraco mas ligado a uma decisão terapêutica explícita; entra como
  fator de risco, não como gatilho isolado.
- **Sinais vitais** (AUC 0,555) — pouco acima do acaso hora a hora; é **triagem para
  revisão humana**, nunca diagnóstico.

Um paciente em que **duas séries independentes** disparam sobe de prioridade: a
corroboração vale mais que qualquer um dos sinais sozinho.

Módulo de biblioteca: o ponto de entrada é ``python -m src.anomaly.cli --alerts``.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import prescriptions, vitals

# Rótulos de prioridade, do mais para o menos urgente.
ALTA, MEDIA, BAIXA = "ALTA", "MEDIA", "BAIXA"

_ORDEM = {ALTA: 0, MEDIA: 1, BAIXA: 2}


@dataclass
class Alerta:
    """Uma linha da fila de plantão."""
    patient: str
    priority: str
    vitals_alerts: int
    last_vitals_hour: int | None
    escalations: int
    last_escalation_hour: int | None
    hours_monitored: int
    source: str


def _prioridade(n_vitais: int, n_escal: int) -> tuple[str, str]:
    """
    Decide prioridade e origem a partir do que disparou.

    A origem acompanha o alerta de propósito: um alerta que não diz o que disparou
    obriga a equipe a reabrir o caso para descobrir, e o custo disso é alto num posto de
    enfermagem.
    """
    if n_vitais and n_escal:
        return ALTA, "vitais + dose"
    if n_escal:
        return MEDIA, "dose"
    if n_vitais:
        return MEDIA, "vitais"
    return BAIXA, "—"


def scan(data_dir: str, limit: int | None = 300,
         detector: vitals.VitalsDetector | None = None) -> pd.DataFrame:
    """
    Varre uma coorte e devolve a fila de plantão.

    Pontua **apenas os pacientes retidos** — os que não participaram do treino. Rodar o
    painel sobre a coorte de treino mostraria o modelo reconhecendo o que memorizou, não
    o que ele faria num plantão real.
    """
    det = detector or vitals.load()
    raw = vitals.load_dataset(data_dir, limit=limit)
    _, teste = vitals.split_cohorts(vitals.prepare(raw))

    pontuado = det.score(teste)
    com_dose = prescriptions.detect(prescriptions.build_series(pontuado))

    linhas: list[Alerta] = []
    for pid, g in com_dose.groupby("patient"):
        alertas_v = g.loc[g["is_anomaly"] == 1, "hour"]
        escal = g.loc[g["is_escalation"] == 1, "hour"]
        prioridade, origem = _prioridade(len(alertas_v), len(escal))
        if prioridade == BAIXA:
            continue                      # a fila só mostra quem tem algo a mostrar
        linhas.append(Alerta(
            patient=str(pid),
            priority=prioridade,
            vitals_alerts=int(len(alertas_v)),
            last_vitals_hour=int(alertas_v.max()) if len(alertas_v) else None,
            escalations=int(len(escal)),
            last_escalation_hour=int(escal.max()) if len(escal) else None,
            hours_monitored=int(len(g)),
            source=origem,
        ))

    df = pd.DataFrame([vars(a) for a in linhas])
    if df.empty:
        return df
    # dentro da mesma prioridade, quem alertou mais recentemente vem primeiro
    df["_ordem"] = df["priority"].map(_ORDEM)
    df["_recente"] = df[["last_vitals_hour", "last_escalation_hour"]].max(axis=1)
    return (df.sort_values(["_ordem", "_recente"], ascending=[True, False])
              .drop(columns=["_ordem", "_recente"])
              .reset_index(drop=True))


def render(df: pd.DataFrame, top: int = 15) -> str:
    """Painel de plantão em texto, no formato de um monitor de posto de enfermagem."""
    if df.empty:
        return "Nenhum alerta ativo na coorte monitorada."

    L = [f"{'=' * 78}",
         f"{'PAINEL DE PLANTÃO — ALERTAS ATIVOS':^78}",
         f"{'=' * 78}",
         ""]

    contagem = df["priority"].value_counts()
    L.append(f"{len(df)} pacientes com alerta  "
             f"(ALTA {contagem.get(ALTA, 0)} · MEDIA {contagem.get(MEDIA, 0)})")
    L.append("")
    L.append(f"{'PRIOR.':<7} {'PACIENTE':<10} {'INTERN.':>8} {'VITAIS':>7} "
             f"{'DOSE':>5} {'ÚLTIMO':>7}  ORIGEM")
    L.append("-" * 78)

    for _, r in df.head(top).iterrows():
        horas = [h for h in (r.last_vitals_hour, r.last_escalation_hour)
                 if h is not None and not pd.isna(h)]
        ult = f"h{int(max(horas))}" if horas else "—"
        L.append(f"{r.priority:<7} {r.patient:<10} "
                 f"{str(r.hours_monitored) + ' h':>8} {r.vitals_alerts:>7} "
                 f"{r.escalations:>5} {ult:>7}  {r.source}")

    if len(df) > top:
        L.append(f"... e mais {len(df) - top} paciente(s)")

    L += ["", "-" * 78,
          "Prioridade derivada da confiabilidade medida de cada modalidade (§5.7):",
          "  movimentação AUC 0,9999 -> automático | vitais AUC 0,555 -> revisão humana",
          "Sinais vitais NÃO constituem diagnóstico: a fila é de triagem.",
          ""]
    return "\n".join(L)
