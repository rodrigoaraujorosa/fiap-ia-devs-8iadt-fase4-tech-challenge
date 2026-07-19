"""
Sinais vitais de UTI — PhysioNet/CinC Challenge 2019 (Sepsis).

Um arquivo ``.psv`` por paciente, uma linha por hora de internação. A detecção é
não-supervisionada (IsolationForest sobre os 8 sinais vitais) e o ``SepsisLabel`` do
dataset entra **só na avaliação**, como ground-truth de deterioração clínica.

Duas decisões importantes de modelagem:

1. **Normalização por paciente.** Um IsolationForest global aprende "o que é raro na
   população", e não "o que mudou neste paciente". Como o alerta de leito é sobre o
   próprio paciente, cada série é convertida em desvio robusto (z-score por MAD) contra
   a mediana das primeiras horas daquele paciente. Sem isso o modelo sinaliza sobretudo
   pacientes cronicamente fora da faixa, não pacientes que estão piorando.

2. **Antecedência (lead time) é a métrica que importa.** Acertar a hora exata em que o
   ``SepsisLabel`` vira 1 é menos útil clinicamente do que alertar antes. Medimos quantas
   horas antes do início da sepse o alerta ocorre, dentro de uma janela de 48 h.

3. **Treino e inferência são separados.** O detector é treinado numa coorte de pacientes
   que nunca desenvolveram sepse, persistido em disco, e depois pontua pacientes que não
   participaram do treino — inclusive um por vez, via ``monitor_patient()``. Sem essa
   separação não haveria detector algum: ajustar o modelo sobre os mesmos dados que se
   quer avaliar mede memorização, não capacidade de alertar.

Módulo de biblioteca: o ponto de entrada é ``python -m src.anomaly.cli``.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# modelo persistido: treinado uma vez, usado para pontuar pacientes novos
MODEL_PATH = Path("models/vitals_detector.joblib")

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]

# Marcadores laboratoriais classicamente associados a sepse. Não entram na detecção
# principal (a subtarefa é sobre sinais vitais), mas servem à comparação medida em
# ``compare_feature_sets()`` — ver §5 do relatório técnico.
SEPSIS_LABS = ["Lactate", "WBC", "Creatinine", "Platelets", "BUN", "pH", "HCO3", "FiO2"]

RANDOM_STATE = 42

# horas iniciais usadas como linha de base do próprio paciente
BASELINE_HOURS = 8


def load_patient(path: str) -> pd.DataFrame:
    """Lê um .psv (pipe-delimited) e acrescenta o id do paciente e a hora."""
    df = pd.read_csv(path, sep="|")
    df["patient"] = os.path.splitext(os.path.basename(path))[0]
    df["hour"] = np.arange(len(df))
    return df


def load_dataset(data_dir: str, limit: int | None = None) -> pd.DataFrame:
    """Carrega os pacientes encontrados em data_dir (recursivo)."""
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.psv"), recursive=True))
    if not files:
        raise FileNotFoundError(f"Nenhum .psv encontrado em {data_dir}")
    if limit:
        files = files[:limit]
    return pd.concat((load_patient(f) for f in files), ignore_index=True)


def _robust_z(group: pd.DataFrame) -> pd.DataFrame:
    """
    Converte os vitais em desvio robusto contra a linha de base do próprio paciente.

    Usa mediana e MAD das primeiras ``BASELINE_HOURS`` horas — mesma técnica de
    ``src/video/anomaly.py``. O MAD resiste a outliers, que é o caso aqui: um único
    registro absurdo não deve definir a escala do paciente.
    """
    base = group[VITALS].head(BASELINE_HOURS)
    # se a linha de base estiver vazia (muito NaN), cai para o paciente inteiro
    med = base.median()
    med = med.fillna(group[VITALS].median())
    mad = (base - med).abs().median()
    mad = mad.fillna((group[VITALS] - med).abs().median())
    # MAD zero (sinal constante na base) viraria divisão por zero
    mad = mad.replace(0, np.nan).fillna(1.0)
    return (group[VITALS] - med) / (1.4826 * mad)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Preenche lacunas por paciente e normaliza contra a linha de base individual."""
    df = df.sort_values(["patient", "hour"]).copy()
    # vitais são medidos de forma esparsa: repete o último valor conhecido dentro do
    # paciente antes de qualquer imputação global
    df[VITALS] = df.groupby("patient")[VITALS].ffill()
    z = df.groupby("patient", group_keys=False)[VITALS + ["patient"]].apply(_robust_z)
    z.columns = [f"z_{c}" for c in VITALS]
    out = pd.concat([df, z], axis=1)
    # o que ainda restou de NaN vira 0 = "sem desvio observado"
    out[[f"z_{c}" for c in VITALS]] = out[[f"z_{c}" for c in VITALS]].fillna(0.0)
    return out


def live_features(df: pd.DataFrame) -> list[str]:
    """
    Colunas z com sinal de fato.

    ``EtCO2`` tem **0% de cobertura** no training set A: a coluna existe no schema, mas
    nunca é medida. Sem esse filtro, uma feature constante em zero entra no modelo e só
    dilui a distância entre as amostras.
    """
    return [f"z_{c}" for c in VITALS if df[f"z_{c}"].abs().sum() > 0]


@dataclass
class VitalsDetector:
    """
    Detector treinado, pronto para pontuar um paciente que o modelo nunca viu.

    Guarda o limiar junto com o modelo. O limiar é **absoluto** — o percentil dos scores
    da coorte de treino — e não um percentil calculado dentro do paciente. A diferença
    importa na hora do alerta de leito: com limiar percentual por paciente, todo paciente
    recebe alertas por construção (sempre existe um "5% pior"), inclusive o paciente
    estável, e estadias curtas não recebem nenhum porque 5% de 19 horas arredonda para
    zero. Com limiar absoluto, um paciente estável pode passar a internação inteira sem
    alerta, que é o comportamento correto.
    """
    model: object
    features: list[str]
    threshold: float
    contamination: float
    trained_on: dict

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pontua uma série já passada por ``prepare()``. Não reaproveita rótulo algum."""
        faltando = [c for c in self.features if c not in df.columns]
        if faltando:
            raise ValueError(f"features ausentes na série: {faltando}")
        out = df.copy()
        out["score"] = self.model.score_samples(df[self.features].to_numpy())
        out["is_anomaly"] = (out["score"] <= self.threshold).astype(int)
        return out


def split_cohorts(df: pd.DataFrame, test_size: float = 0.3,
                  seed: int = RANDOM_STATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separa treino e teste **por paciente**, nunca por hora.

    O treino recebe apenas pacientes que **nunca desenvolveram sepse** — é o padrão de
    normalidade que o detector deve aprender. Manter pacientes sépticos no treino
    ensinaria ao modelo que a deterioração é normal, que é exatamente o oposto do
    objetivo. Mesmo desenho já usado na movimentação, onde o modelo só vê repouso.

    O teste reúne os pacientes com sepse (todos) e os pacientes sem sepse retidos, para
    que a avaliação tenha as duas classes.
    """
    rng = np.random.default_rng(seed)
    por_paciente = df.groupby("patient")["SepsisLabel"].max()
    sem_sepse = por_paciente[por_paciente == 0].index.to_numpy()
    com_sepse = por_paciente[por_paciente == 1].index.to_numpy()

    rng.shuffle(sem_sepse)
    corte = int(len(sem_sepse) * (1 - test_size))
    treino_ids, retidos_ids = sem_sepse[:corte], sem_sepse[corte:]

    treino = df[df["patient"].isin(treino_ids)]
    teste = df[df["patient"].isin(np.concatenate([retidos_ids, com_sepse]))]
    return treino, teste


def fit(df_train: pd.DataFrame, contamination: float = 0.05) -> VitalsDetector:
    """
    Treina o detector na coorte de normalidade e fixa o limiar de alerta.

    ``df_train`` já deve ter passado por ``prepare()``. O limiar é o percentil
    ``contamination`` dos scores do próprio treino: por construção, o detector alerta em
    cerca de 5% das horas de um paciente **estável**, e mais do que isso num paciente
    que se afasta desse padrão.
    """
    from sklearn.ensemble import IsolationForest

    cols = live_features(df_train)
    X = df_train[cols].to_numpy()

    model = IsolationForest(contamination=contamination, random_state=RANDOM_STATE)
    model.fit(X)

    scores = model.score_samples(X)
    threshold = float(np.quantile(scores, contamination))

    return VitalsDetector(
        model=model,
        features=cols,
        threshold=threshold,
        contamination=contamination,
        trained_on={
            "patients": int(df_train["patient"].nunique()),
            "hours": int(len(df_train)),
            "sepsis_patients": int(df_train.groupby("patient")["SepsisLabel"].max().sum()),
        },
    )


def save(detector: VitalsDetector, path: Path = MODEL_PATH) -> Path:
    import joblib
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(detector, path)
    return path


def load(path: Path = MODEL_PATH) -> VitalsDetector:
    import joblib
    if not path.exists():
        raise FileNotFoundError(
            f"nenhum modelo em {path} — treine antes: python -m src.anomaly.cli --train")
    return joblib.load(path)


def evaluate(df: pd.DataFrame) -> dict:
    """
    Avalia o alerta hora-a-hora contra o SepsisLabel.

    Nota importante para a leitura dos números: o SepsisLabel marca a janela em que a
    sepse é considerada instalada, e não "hora anormal". Um alerta fora dessa janela não
    é necessariamente falso — pode ser instabilidade real que não evoluiu para sepse.
    Por isso a precisão hora-a-hora é conservadora por construção.
    """
    from sklearn.metrics import (average_precision_score,
                                 precision_recall_fscore_support, roc_auc_score)

    y_true = df["SepsisLabel"]
    y_pred = df["is_anomaly"]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)

    return {
        "rows": len(df),
        "patients": int(df["patient"].nunique()),
        "sepsis_rows": int(y_true.sum()),
        "sepsis_patients": int(df.groupby("patient")["SepsisLabel"].max().sum()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y_true, -df["score"])),
        # com ~2,4% de prevalência a AUPRC diz mais que a AUC; o valor só tem
        # significado comparado à prevalência, que é o acerto de um sorteio
        "auprc": float(average_precision_score(y_true, -df["score"])),
        "prevalence": float(y_true.mean()),
        "alert_rate": float(y_pred.mean()),
    }


def compare_feature_sets(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    """
    Compara o poder de discriminação dos sinais vitais e dos marcadores de laboratório.

    Resultado medido (300 pacientes do training set A): os labs discriminam melhor que
    os vitais **apesar de terem cobertura muito menor** (4-14% contra 83-91%). É
    coerente com a clínica — lactato e leucócitos são marcadores diretos de sepse,
    enquanto os vitais reagem tarde. Está na §5 do relatório técnico como limitação
    medida da subtarefa, não como troca de escopo: a entrega é sobre sinais vitais.
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = df["SepsisLabel"]
    vitais = live_features(df)

    # labs: ffill dentro do paciente e escala robusta global — não há linha de base
    # individual confiável, porque muitos pacientes só têm uma ou duas coletas
    lab = df.groupby("patient")[SEPSIS_LABS].ffill()
    iqr = (lab.quantile(0.75) - lab.quantile(0.25)).replace(0, 1)
    lab = ((lab - lab.median()) / iqr).fillna(0.0).add_prefix("L_")
    d = pd.concat([df, lab], axis=1)

    linhas = []
    for nome, feat in [("sinais vitais", vitais),
                       ("marcadores de laboratório", list(lab.columns)),
                       ("vitais + laboratório", vitais + list(lab.columns))]:
        model = IsolationForest(contamination=contamination, random_state=RANDOM_STATE)
        score = -model.fit(d[feat]).score_samples(d[feat])
        linhas.append({
            "features": nome,
            "n_features": len(feat),
            "roc_auc": round(float(roc_auc_score(y, score)), 4),
            "auprc": round(float(average_precision_score(y, score)), 4),
        })
    return pd.DataFrame(linhas)


# Janela em que um alerta ainda conta como aviso prévio do evento. Alertas muito antes
# do início da sepse não são aviso: são instabilidade não relacionada.
LEAD_WINDOW_HOURS = 48


def lead_time(df: pd.DataFrame, window: int = LEAD_WINDOW_HOURS) -> pd.DataFrame:
    """
    Antecedência do alerta em relação ao início da sepse, por paciente.

    **Só contam alertas dentro da janela de ``window`` horas antes do início.** Medir a
    partir do *primeiro alerta de toda a internação* infla o resultado sem significado
    clínico: o paciente ``p000009`` tem sepse na hora 248 e alertas nas primeiras 60
    horas, o que produziria uma "antecedência" de 239 horas para um alerta que nada tem
    a ver com o evento. Dentro da janela, a pergunta passa a ser a que importa — o
    sistema avisou a tempo de agir?

    Colunas devolvidas:
      - ``alert_in_window``: houve alerta na janela;
      - ``lead_hours``: horas entre o alerta mais antigo *da janela* e o início;
      - ``alerts_before_onset``: total de alertas antes do início (contexto);
      - ``any_alert``: o paciente recebeu algum alerta na internação inteira.
    """
    linhas = []
    for pid, g in df.groupby("patient"):
        if g["SepsisLabel"].max() != 1:
            continue
        onset = int(g.loc[g["SepsisLabel"] == 1, "hour"].min())
        alertas = g.loc[g["is_anomaly"] == 1, "hour"]
        antes = alertas[alertas < onset]
        na_janela = antes[antes >= onset - window]

        linhas.append({
            "patient": pid,
            "onset_hour": onset,
            "any_alert": bool(len(alertas)),
            "alerts_before_onset": int(len(antes)),
            "alert_in_window": bool(len(na_janela)),
            "first_alert_in_window": int(na_janela.min()) if len(na_janela) else None,
            "lead_hours": onset - int(na_janela.min()) if len(na_janela) else None,
        })
    return pd.DataFrame(linhas)


def monitor_patient(patient_id: str, data_dir: str,
                    detector: VitalsDetector | None = None) -> dict:
    """
    Modo de inferência: pontua **um** paciente com o modelo já treinado.

    É o caminho que o monitoramento de leito percorreria — o modelo não conhece este
    paciente, recebe a série dele e devolve as horas de alerta. O ``SepsisLabel`` é lido
    apenas para conferência posterior, nunca entra na pontuação.
    """
    det = detector or load()
    caminhos = glob.glob(os.path.join(data_dir, "**", f"{patient_id}.psv"), recursive=True)
    if not caminhos:
        raise FileNotFoundError(f"paciente {patient_id} não encontrado em {data_dir}")

    serie = prepare(load_patient(caminhos[0]))
    pontuado = det.score(serie)
    alertas = pontuado[pontuado["is_anomaly"] == 1]

    resumo = {
        "patient": patient_id,
        "hours": len(pontuado),
        "alerts": int(len(alertas)),
        "alert_hours": alertas["hour"].tolist(),
        "alert_rate": float(pontuado["is_anomaly"].mean()),
        "developed_sepsis": bool(pontuado["SepsisLabel"].max() == 1),
    }
    if resumo["developed_sepsis"]:
        onset = int(pontuado.loc[pontuado["SepsisLabel"] == 1, "hour"].min())
        na_janela = [h for h in resumo["alert_hours"]
                     if onset - LEAD_WINDOW_HOURS <= h < onset]
        resumo.update({
            "onset_hour": onset,
            "warned": bool(na_janela),
            "lead_hours": onset - min(na_janela) if na_janela else None,
        })
    return {"data": pontuado, "summary": resumo}


def run(data_dir: str, limit: int | None = None, contamination: float = 0.05) -> dict:
    """
    Treina na coorte de normalidade e avalia nos pacientes retidos.

    Nenhum paciente do teste aparece no treino, e nenhum paciente com sepse é usado para
    treinar. As métricas abaixo são, portanto, de **generalização** — o que o detector
    faz com séries que nunca viu.
    """
    raw = load_dataset(data_dir, limit=limit)
    preparado = prepare(raw)

    treino, teste = split_cohorts(preparado)
    detector = fit(treino, contamination=contamination)
    df = detector.score(teste)

    metrics = evaluate(df)
    metrics.update({
        "train_patients": detector.trained_on["patients"],
        "train_hours": detector.trained_on["hours"],
        "test_patients": int(teste["patient"].nunique()),
        "features_used": len(detector.features),
        "threshold": detector.threshold,
    })
    lt = lead_time(df)
    na_janela = lt[lt["alert_in_window"]] if len(lt) else lt

    metrics.update({
        "contamination": contamination,
        "coverage": {c: float(raw[c].notna().mean()) for c in VITALS},
        "lead_window_hours": LEAD_WINDOW_HOURS,
        # quantos pacientes com sepse receberam qualquer alerta na internação
        "sepsis_patients_alerted": int(lt["any_alert"].sum()) if len(lt) else 0,
        # e quantos foram avisados a tempo — este é o número que importa
        "sepsis_patients_warned": int(len(na_janela)),
        "lead_median_hours": float(na_janela["lead_hours"].median())
                             if len(na_janela) else None,
        "lead_mean_hours": float(na_janela["lead_hours"].mean())
                           if len(na_janela) else None,
    })
    return {"data": df, "metrics": metrics, "lead_time": lt, "detector": detector,
            "prepared": preparado}


def plot_patient(df: pd.DataFrame, patient: str, out_path: str) -> str | None:
    """Figura: série de vitais de um paciente, com alertas e o início da sepse."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = df[df["patient"] == patient]
    if sub.empty:
        return None

    fig, ax = plt.subplots(figsize=(11, 5))
    for v in ["HR", "SBP", "Resp", "O2Sat"]:
        ax.plot(sub["hour"], sub[v], label=v, alpha=0.85, linewidth=1.3)

    anom = sub[sub["is_anomaly"] == 1]
    ax.scatter(anom["hour"], anom["HR"], color="#c0392b", zorder=5, s=28,
               label="hora sinalizada")

    if sub["SepsisLabel"].max() == 1:
        onset = sub.loc[sub["SepsisLabel"] == 1, "hour"].min()
        ax.axvline(onset, color="#8e44ad", linestyle="--", linewidth=1.8,
                   label="início da sepse (ground-truth)")

    ax.set_xlabel("hora de internação")
    ax.set_ylabel("valor medido")
    ax.set_title(f"Sinais vitais e alertas — paciente {patient}")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
