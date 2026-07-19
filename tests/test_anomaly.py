"""
Testes da Entrega 3 com séries sintéticas (sem baixar Challenge 2019 nem UCI HAR).

Cobrem sobretudo as duas armadilhas encontradas ao rodar com dados reais, que são
silenciosas — não levantam erro, só produzem números bonitos e errados:

1. **Lead time sem janela.** Medir a antecedência a partir do primeiro alerta da
   internação inteira transforma um alerta 240 horas antes do evento em "aviso prévio".
2. **Limiar global x por paciente.** Um limiar global gasta o orçamento de alertas nos
   pacientes cronicamente instáveis e deixa os demais sem nenhum alerta.
"""
import pathlib

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


def _coorte(n_estaveis: int = 12, n_sepse: int = 4, horas: int = 120) -> pd.DataFrame:
    partes = [_patient(f"ok{i}", horas) for i in range(n_estaveis)]
    partes += [_patient(f"sep{i}", horas, onset=90) for i in range(n_sepse)]
    return vitals.prepare(pd.concat(partes, ignore_index=True))


def test_split_nao_vaza_paciente_entre_treino_e_teste():
    """Split por paciente, nunca por hora — senão o mesmo paciente treina e testa."""
    treino, teste = vitals.split_cohorts(_coorte())
    assert set(treino["patient"]) & set(teste["patient"]) == set()


def test_treino_nao_contem_paciente_com_sepse():
    """
    A coorte de treino é o padrão de normalidade.

    Deixar pacientes sépticos no treino ensina ao modelo que a deterioração é normal —
    o oposto do objetivo do detector.
    """
    treino, teste = vitals.split_cohorts(_coorte())
    assert treino.groupby("patient")["SepsisLabel"].max().sum() == 0
    # e todos os pacientes com sepse precisam estar disponíveis para avaliar
    assert teste.groupby("patient")["SepsisLabel"].max().sum() == 4


def test_limiar_absoluto_separa_estavel_de_deteriorando():
    """
    Diferença central do limiar absoluto para o percentual por paciente.

    Com corte percentual dentro do paciente sempre existe um "5% pior", então **todo**
    paciente recebe alerta por construção, inclusive o estável — o que torna a taxa de
    alerta incomparável entre pacientes. Com limiar absoluto, a taxa passa a significar
    alguma coisa: o paciente estável fica na taxa-base do treino (~contamination) e o
    que deteriora fica claramente acima.

    Note que o estável **não** fica em zero, e não deveria: o limiar é o percentil 5 dos
    scores de treino, então um paciente estatisticamente igual ao treino dispara em
    torno de 5% das horas. O que o teste fixa é a separação entre os dois.
    """
    treino, _ = vitals.split_cohorts(_coorte(n_estaveis=16, n_sepse=2))
    det = vitals.fit(treino, contamination=0.05)

    estavel = det.score(vitals.prepare(_patient("novo_estavel", 120)))
    grave = det.score(vitals.prepare(_patient("novo_grave", 120, onset=90)))

    taxa_estavel = estavel["is_anomaly"].mean()
    taxa_grave = grave["is_anomaly"].mean()

    assert taxa_estavel <= 0.10, "paciente estável não pode viver em alerta"
    assert taxa_grave > taxa_estavel, "quem deteriora tem de alertar mais que o estável"


def test_alerta_do_paciente_grave_se_concentra_na_deterioracao():
    """O alerta tem de cair perto do evento, não espalhado pela internação."""
    treino, _ = vitals.split_cohorts(_coorte(n_estaveis=16, n_sepse=2))
    det = vitals.fit(treino, contamination=0.05)

    grave = det.score(vitals.prepare(_patient("novo_grave", 120, onset=90)))
    alertas = grave.loc[grave["is_anomaly"] == 1, "hour"]
    assert len(alertas) > 0
    # a deterioração sintética começa 6 h antes do onset (hora 84)
    assert ((alertas >= 80) & (alertas < 90)).any()


def test_modelo_salvo_e_recarregado_pontua_igual(tmp_path):
    """O modelo persistido tem de produzir exatamente o mesmo alerta."""
    treino, teste = vitals.split_cohorts(_coorte())
    det = vitals.fit(treino)

    caminho = vitals.save(det, tmp_path / "detector.joblib")
    recarregado = vitals.load(caminho)

    assert recarregado.threshold == det.threshold
    assert recarregado.features == det.features
    pd.testing.assert_series_equal(det.score(teste)["is_anomaly"],
                                   recarregado.score(teste)["is_anomaly"])


def test_load_sem_modelo_orienta_a_treinar():
    with pytest.raises(FileNotFoundError, match="--train"):
        vitals.load(pathlib.Path("nao/existe/detector.joblib"))


def test_score_nao_usa_o_rotulo():
    """
    O SepsisLabel não pode influenciar a pontuação.

    Embaralhar o rótulo tem de deixar os alertas idênticos — se mudar, o rótulo vazou
    para dentro do modelo.
    """
    treino, teste = vitals.split_cohorts(_coorte())
    det = vitals.fit(treino)

    original = det.score(teste)["is_anomaly"].reset_index(drop=True)
    embaralhado = teste.copy()
    embaralhado["SepsisLabel"] = RNG.permutation(embaralhado["SepsisLabel"].values)
    depois = det.score(embaralhado)["is_anomaly"].reset_index(drop=True)

    pd.testing.assert_series_equal(original, depois)


def test_evaluate_devolve_metricas_coerentes():
    treino, teste = vitals.split_cohorts(_coorte())
    det = vitals.fit(treino)
    m = vitals.evaluate(det.score(teste))
    assert 0.0 <= m["roc_auc"] <= 1.0
    assert 0.0 <= m["auprc"] <= 1.0
    assert m["sepsis_patients"] == 4


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


def test_monitor_de_paciente_sem_dose_nao_entra_em_monitoramento():
    """Quem tem menos de 3 registros de FiO2 não está sob suporte de oxigênio."""
    df = _patient("p1", 40)                       # FiO2 é NaN em _patient()
    s = prescriptions.monitor_patient(df)["summary"]
    assert not s["monitored"]
    assert s["observations"] == 0


def test_monitor_de_paciente_reporta_escalonamentos_e_antecedencia():
    """Escalonamento na hora 5, sepse na hora 40 -> fora da janela; na 20 -> dentro."""
    doses = [0.30] * 20 + [0.60] * 30
    s = prescriptions.monitor_patient(_com_dose("p1", doses, onset=40))["summary"]
    assert s["monitored"]
    assert s["escalations"] == 1
    assert s["escalation_hours"] == [20]
    assert s["lead_hours"] == 20


def test_prescricoes_lead_time_respeita_a_janela():
    """Mesma regra dos vitais: escalonamento antigo demais não conta como aviso."""
    doses = [0.30] * 5 + [0.60] * 195
    df = prescriptions.detect(prescriptions.build_series(
        _com_dose("p1", doses, onset=190)))
    assert len(prescriptions.lead_time(df, window=48)) == 0      # escalonou na hora 5
