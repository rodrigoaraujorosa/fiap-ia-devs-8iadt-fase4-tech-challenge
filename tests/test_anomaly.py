"""
Testes da Entrega 3 com séries sintéticas (sem baixar Challenge 2019 nem UCI HAR).

Cobrem sobretudo as duas armadilhas encontradas ao rodar com dados reais, que são
silenciosas — não levantam erro, só produzem números bonitos e errados:

1. **Lead time sem janela.** Medir a antecedência a partir do primeiro alerta da
   internação inteira transforma um alerta 240 horas antes do evento em "aviso prévio".
2. **Limiar global x por paciente.** Um limiar global gasta o orçamento de alertas nos
   pacientes cronicamente instáveis e deixa os demais sem nenhum alerta.
"""
import numpy as np
import pandas as pd
import pytest

from src.anomaly import prescriptions, vitals

RNG = np.random.default_rng(42)


def _patient(pid: str, hours: int, onset: int | None = None,
             unstable: bool = False) -> pd.DataFrame:
    """
    Série sintética de um paciente.

    ``unstable=True`` injeta **excursões** frequentes — picos grandes e esparsos, não
    ruído de maior amplitude. A distinção importa: a normalização por MAD equaliza
    diferenças de escala entre pacientes, então ruído maior sozinho não concentra
    alertas. O que consome o orçamento global é o paciente que sai repetidamente da
    própria linha de base, e é esse o caso observado nos dados reais.
    """
    escala = 2.0
    df = pd.DataFrame({
        "patient": pid,
        "hour": np.arange(hours),
        "HR": 80 + RNG.normal(0, escala, hours),
        "O2Sat": 97 + RNG.normal(0, escala / 4, hours),
        "Temp": 37 + RNG.normal(0, escala / 12, hours),
        "SBP": 120 + RNG.normal(0, escala, hours),
        "MAP": 85 + RNG.normal(0, escala, hours),
        "DBP": 70 + RNG.normal(0, escala, hours),
        "Resp": 16 + RNG.normal(0, escala / 4, hours),
        "EtCO2": np.nan,                     # como no dataset real: 0% de cobertura
        "FiO2": np.nan,
        "SepsisLabel": 0,
    })
    if unstable:
        # excursões em ~25% das horas, muito além da própria linha de base
        picos = RNG.choice(np.arange(vitals.BASELINE_HOURS, hours),
                           size=max(1, hours // 4), replace=False)
        df.loc[df.hour.isin(picos), "HR"] += 60
        df.loc[df.hour.isin(picos), "SBP"] += 50
        df.loc[df.hour.isin(picos), "Resp"] += 14
    if onset is not None:
        df.loc[df.hour >= onset, "SepsisLabel"] = 1
        # deterioração nas horas que antecedem o início
        janela = (df.hour >= onset - 6) & (df.hour < onset)
        df.loc[janela, "HR"] += 45
        df.loc[janela, "SBP"] -= 30
        df.loc[janela, "O2Sat"] -= 12
    return df


def test_live_features_descarta_coluna_sem_cobertura():
    """EtCO2 é 100% NaN e não pode entrar no modelo como constante zero."""
    df = vitals.prepare(_patient("p1", 60))
    assert "z_EtCO2" not in vitals.live_features(df)
    assert "z_HR" in vitals.live_features(df)


def test_lead_time_ignora_alerta_fora_da_janela():
    """
    Alerta muito antes do evento não é aviso prévio.

    Este é o teste da armadilha: sepse na hora 200, único alerta na hora 5. Sem janela,
    o cálculo devolveria 195 horas de "antecedência".
    """
    df = _patient("p1", 220, onset=200)
    df["is_anomaly"] = 0
    df.loc[df.hour == 5, "is_anomaly"] = 1

    lt = vitals.lead_time(df, window=48)
    linha = lt.iloc[0]
    assert linha["any_alert"] is True or linha["any_alert"] == 1
    assert not linha["alert_in_window"]
    assert linha["lead_hours"] is None or pd.isna(linha["lead_hours"])


def test_lead_time_conta_alerta_dentro_da_janela():
    df = _patient("p1", 220, onset=200)
    df["is_anomaly"] = 0
    df.loc[df.hour == 180, "is_anomaly"] = 1      # 20 h antes do início

    lt = vitals.lead_time(df, window=48)
    assert bool(lt.iloc[0]["alert_in_window"])
    assert lt.iloc[0]["lead_hours"] == 20


def test_lead_time_ignora_alerta_posterior_ao_inicio():
    """Alerta depois do evento não é antecedência."""
    df = _patient("p1", 220, onset=200)
    df["is_anomaly"] = 0
    df.loc[df.hour == 210, "is_anomaly"] = 1

    lt = vitals.lead_time(df, window=48)
    assert not lt.iloc[0]["alert_in_window"]


def test_limiar_por_paciente_alcanca_todos_os_pacientes():
    """
    O modo por paciente não deixa um paciente instável consumir todos os alertas.

    Com limiar global, o paciente ruidoso concentra os alertas e os estáveis ficam sem
    nenhum; por paciente, cada um é vigiado contra a própria linha de base.
    """
    df = pd.concat([
        _patient("ruidoso", 120, unstable=True),
        _patient("estavel_a", 120),
        _patient("estavel_b", 120),
    ], ignore_index=True)
    prep = vitals.prepare(df)

    por_paciente = vitals.detect(prep, contamination=0.05, per_patient=True)
    alcancados = por_paciente.groupby("patient")["is_anomaly"].max()
    assert alcancados.all(), "todo paciente com estadia longa deve receber algum alerta"

    globais = vitals.detect(prep, contamination=0.05, per_patient=False)
    do_ruidoso = globais.loc[globais.patient == "ruidoso", "is_anomaly"].sum()
    total = globais["is_anomaly"].sum()
    assert do_ruidoso > total / 2, "o cenário deve mesmo concentrar alertas no ruidoso"


def test_estadia_curta_nao_rende_alerta_percentual():
    """
    Limitação conhecida e documentada: 5% de 19 horas arredonda para zero alertas.

    O teste existe para que a limitação seja notada se alguém trocar o critério.
    """
    df = vitals.prepare(pd.concat([_patient("curto", 19), _patient("longo", 200)],
                                  ignore_index=True))
    res = vitals.detect(df, contamination=0.05, per_patient=True)
    assert res.loc[res.patient == "curto", "is_anomaly"].sum() == 0
    assert res.loc[res.patient == "longo", "is_anomaly"].sum() > 0


def test_evaluate_devolve_metricas_coerentes():
    df = vitals.prepare(pd.concat(
        [_patient(f"p{i}", 120, onset=90 if i < 3 else None) for i in range(8)],
        ignore_index=True))
    m = vitals.evaluate(vitals.detect(df))
    assert 0.0 <= m["roc_auc"] <= 1.0
    assert 0.0 <= m["auprc"] <= 1.0
    assert m["patients"] == 8
    assert m["sepsis_patients"] == 3


# ------------------------------------------------------------------ prescrições

def _com_dose(pid: str, doses: list[float], onset: int | None = None) -> pd.DataFrame:
    df = _patient(pid, len(doses), onset=onset)
    df["FiO2"] = doses
    return df


def test_escalonamento_detectado_e_desmame_nao():
    """Subir a dose alerta; reduzir é desmame e não pode alertar."""
    doses = [0.30] * 5 + [0.60] * 5 + [0.30] * 5      # sobe 0,30 e depois cai 0,30
    r = prescriptions.detect(prescriptions.build_series(_com_dose("p1", doses)))
    assert r["is_escalation"].sum() == 1
    assert r["is_weaning"].sum() == 1
    assert r.loc[r["is_escalation"] == 1, "hour"].iloc[0] == 5


def test_dose_em_percentual_normalizada_para_fracao():
    """O dataset mistura escalas (21-100 e 0,21-1,0); ambas viram fração."""
    r = prescriptions.build_series(_com_dose("p1", [50.0] * 4))
    assert r["dose"].max() == pytest.approx(0.50)


def test_variacao_abaixo_do_limiar_nao_alerta():
    doses = [0.40] * 5 + [0.50] * 5                   # degrau de 0,10 < 0,15
    r = prescriptions.detect(prescriptions.build_series(_com_dose("p1", doses)))
    assert r["is_escalation"].sum() == 0


def test_prescricoes_lead_time_respeita_a_janela():
    """Mesma regra dos vitais: escalonamento antigo demais não conta como aviso."""
    doses = [0.30] * 5 + [0.60] * 195
    df = prescriptions.detect(prescriptions.build_series(
        _com_dose("p1", doses, onset=190)))
    assert len(prescriptions.lead_time(df, window=48)) == 0      # escalonou na hora 5
