"""
Evolução de prescrições — variável derivada do PhysioNet Challenge 2019.

**Por que derivada, e não uma base de prescrições.** Não existe fonte pública aberta e
granular de prescrições hospitalares: a base de referência é o MIMIC-IV, que exige curso
CITI e Data Use Agreement (semanas de tramitação). As duas saídas defensáveis eram gerar
dados sintéticos com o Synthea ou derivar a série de uma variável de intervenção já
presente no Challenge 2019. O grupo optou pela segunda — mantém uma única fonte de dados
na entrega e preserva o mesmo ground-truth (``SepsisLabel``).

**A variável escolhida é a FiO2** (fração inspirada de oxigênio). Ao contrário dos demais
campos do dataset, que são *medições* do paciente, a FiO2 é um **valor prescrito**: a
equipe decide e titula a oferta de oxigênio. Sua série ao longo das horas é, portanto,
uma série de doses — que é exatamente o objeto da subtarefa. Também é o campo de melhor
cobertura entre os não-vitais (14,4%).

**O que é anomalia aqui:** um degrau brusco na dose. Escalonar a FiO2 rapidamente
significa que o paciente está precisando de mais suporte respiratório — o equivalente,
na prática clínica, a uma mudança abrupta de prescrição.

Ressalva a declarar na banca: é uma *proxy* de prescrição, não a prescrição registrada em
prontuário. A limitação está na §5 do relatório técnico.

Módulo de biblioteca: o ponto de entrada é ``python -m src.anomaly.cli``.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

# FiO2 é o valor prescrito; O2Sat é a resposta do paciente à dose, usada só no gráfico.
DOSE = "FiO2"
RESPONSE = "O2Sat"

# Degrau considerado relevante: a FiO2 vai de 0,21 (ar ambiente) a 1,0. Uma variação de
# 0,15 entre coletas consecutivas equivale a subir cerca de dois níveis de suporte.
STEP_THRESHOLD = 0.15

# horas mínimas com dose registrada para o paciente entrar na análise
MIN_OBSERVATIONS = 3


def build_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constrói a série de doses por paciente a partir do dataframe já carregado.

    A FiO2 é registrada de forma esparsa; entre duas coletas a prescrição vigente é a
    última informada, então o ``ffill`` dentro do paciente é a leitura correta — não é
    imputação estatística, é o valor que estava valendo.
    """
    out = df.sort_values(["patient", "hour"]).copy()
    out["dose"] = out.groupby("patient")[DOSE].ffill()

    # o dataset mistura escalas: alguns registros usam percentual (21-100) e outros
    # fração (0,21-1,0). Normaliza tudo para fração.
    out.loc[out["dose"] > 1.5, "dose"] = out.loc[out["dose"] > 1.5, "dose"] / 100.0

    out["dose_change"] = out.groupby("patient")["dose"].diff()
    return out


def detect(df: pd.DataFrame, threshold: float = STEP_THRESHOLD) -> pd.DataFrame:
    """
    Sinaliza escalonamentos bruscos da dose.

    Só **aumentos** contam como alerta: reduzir a FiO2 é desmame, sinal de melhora, e
    marcá-lo como anomalia encheria o alerta de falsos positivos benignos.
    """
    out = df.copy()
    out["is_escalation"] = (out["dose_change"] >= threshold).astype(int)
    out["is_weaning"] = (out["dose_change"] <= -threshold).astype(int)
    return out


def evaluate(df: pd.DataFrame) -> dict:
    """
    Avalia o escalonamento de dose contra o SepsisLabel.

    Restringe a análise aos pacientes com dose registrada: quem nunca teve FiO2 medida
    não está sob suporte de oxigênio, e incluí-lo como "sem alerta" mediria a cobertura
    do dataset, não a qualidade do detector.
    """
    obs = df.groupby("patient")["dose"].count()
    elegiveis = obs[obs >= MIN_OBSERVATIONS].index
    sub = df[df["patient"].isin(elegiveis)]

    if sub.empty:
        return {"eligible_patients": 0}

    por_paciente = sub.groupby("patient").agg(
        escalated=("is_escalation", "max"),
        sepsis=("SepsisLabel", "max"),
    )
    tp = int(((por_paciente.escalated == 1) & (por_paciente.sepsis == 1)).sum())
    fp = int(((por_paciente.escalated == 1) & (por_paciente.sepsis == 0)).sum())
    fn = int(((por_paciente.escalated == 0) & (por_paciente.sepsis == 1)).sum())
    tn = int(((por_paciente.escalated == 0) & (por_paciente.sepsis == 0)).sum())

    return {
        "eligible_patients": int(len(elegiveis)),
        "patients_total": int(df["patient"].nunique()),
        "dose_coverage": float(df[DOSE].notna().mean()),
        "escalations": int(sub["is_escalation"].sum()),
        "weanings": int(sub["is_weaning"].sum()),
        "threshold": STEP_THRESHOLD,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "sepsis_rate_escalated": (tp / (tp + fp)) if (tp + fp) else 0.0,
        "sepsis_rate_not_escalated": (fn / (fn + tn)) if (fn + tn) else 0.0,
    }


def lead_time(df: pd.DataFrame, window: int = 48) -> pd.DataFrame:
    """
    Antecedência do escalonamento em relação ao início da sepse.

    Mesma regra de janela usada em ``vitals.lead_time``: um escalonamento de dose 200
    horas antes do evento não é aviso prévio, é outra ocorrência clínica. Só contam os
    escalonamentos nas ``window`` horas que antecedem o início.
    """
    linhas = []
    for pid, g in df.groupby("patient"):
        if g["SepsisLabel"].max() != 1:
            continue
        onset = int(g.loc[g["SepsisLabel"] == 1, "hour"].min())
        esc = g.loc[g["is_escalation"] == 1, "hour"]
        antes = esc[esc < onset]
        na_janela = antes[antes >= onset - window]
        if na_janela.empty:
            continue
        linhas.append({"patient": pid, "onset_hour": onset,
                       "first_escalation_hour": int(na_janela.min()),
                       "lead_hours": onset - int(na_janela.min())})
    return pd.DataFrame(linhas)


def monitor_patient(df_patient: pd.DataFrame,
                    threshold: float = STEP_THRESHOLD) -> dict:
    """
    Modo de inferência: escalonamentos de dose de **um** paciente.

    Não há modelo a carregar — a detecção é uma regra de degrau, e a regra é a mesma
    para um paciente ou para a coorte inteira. Recebe a série já carregada para compor
    com o monitoramento de vitais do mesmo paciente, que vem do mesmo dataset.
    """
    serie = detect(build_series(df_patient), threshold=threshold)
    com_dose = serie.dropna(subset=["dose"])
    esc = serie[serie["is_escalation"] == 1]

    resumo = {
        "observations": int(len(com_dose)),
        "monitored": len(com_dose) >= MIN_OBSERVATIONS,
        "escalations": int(len(esc)),
        "escalation_hours": esc["hour"].tolist(),
        "weanings": int(serie["is_weaning"].sum()),
        "dose_min": float(com_dose["dose"].min()) if len(com_dose) else None,
        "dose_max": float(com_dose["dose"].max()) if len(com_dose) else None,
    }
    if serie["SepsisLabel"].max() == 1 and len(esc):
        onset = int(serie.loc[serie["SepsisLabel"] == 1, "hour"].min())
        na_janela = [h for h in resumo["escalation_hours"] if onset - 48 <= h < onset]
        resumo.update({"onset_hour": onset,
                       "lead_hours": onset - min(na_janela) if na_janela else None})
    return {"data": serie, "summary": resumo}


def run(df: pd.DataFrame, threshold: float = STEP_THRESHOLD) -> dict:
    """Recebe o dataframe já carregado pelo módulo de vitais e roda a subtarefa."""
    serie = detect(build_series(df), threshold=threshold)
    metrics = evaluate(serie)
    lt = lead_time(serie)
    metrics.update({
        "lead_window_hours": 48,
        "lead_median_hours": float(lt["lead_hours"].median()) if len(lt) else None,
        "escalated_in_window": int(len(lt)),
    })
    return {"data": serie, "metrics": metrics, "lead_time": lt}


def plot_patient(df: pd.DataFrame, patient: str, out_path: str) -> str | None:
    """Figura: dose prescrita (FiO2) e resposta (O2Sat), com os escalonamentos."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = df[df["patient"] == patient].dropna(subset=["dose"])
    if sub.empty:
        return None

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.step(sub["hour"], sub["dose"], where="post", color="#2e86c1",
            linewidth=2, label=f"{DOSE} — dose prescrita")

    esc = sub[sub["is_escalation"] == 1]
    ax.scatter(esc["hour"], esc["dose"], color="#c0392b", zorder=5, s=60,
               marker="^", label="escalonamento brusco")

    if sub["SepsisLabel"].max() == 1:
        onset = sub.loc[sub["SepsisLabel"] == 1, "hour"].min()
        ax.axvline(onset, color="#8e44ad", linestyle="--", linewidth=1.8,
                   label="início da sepse (ground-truth)")

    ax.set_xlabel("hora de internação")
    ax.set_ylabel("FiO2 (fração)")
    ax.set_title(f"Evolução da dose prescrita — paciente {patient}")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
