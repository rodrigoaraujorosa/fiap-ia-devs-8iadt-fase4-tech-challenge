"""
App web (Gradio) para demonstração local da Entrega 3 — Detecção de Anomalias.

É o **fluxo final do alerta à equipe médica**: a fila de plantão à esquerda, ordenada por
prioridade, e a série temporal do paciente selecionado à direita. Clicar numa linha da
fila é o gesto do plantonista — ver por que aquele paciente subiu na lista.

A coorte é pontuada **uma vez** e guardada em estado; abrir um paciente não relê os
arquivos. Sem isso, cada clique custaria os ~11 s da varredura.

Uso:
    python -m src.anomaly.app        # abre em http://localhost:7861

Roda inteiramente local, sem custo. Exige o modelo já treinado:
    python -m src.anomaly.cli --train --limit 5000
"""
from __future__ import annotations

from pathlib import Path

import gradio as gr
import pandas as pd

from . import alerts, movement, prescriptions, vitals
from .report import FIGURES_DIR, plot_monitor, plot_subject_timeline

DEFAULT_VITALS_DIR = "data/anomaly/challenge2019"
DEFAULT_HAR_DIR = "data/anomaly/uci_har/UCI HAR Dataset"
DEFAULT_LIMIT = 1000

# Sujeitos que o UCI HAR reserva para teste — o modelo de movimentação não os viu.
HAR_TEST_SUBJECTS = [2, 4, 9, 10, 12, 13, 18, 20, 24]

# A porta 7860 é do app da Entrega 1; os dois podem ficar abertos lado a lado na demo.
PORT = 7861

# As siglas são as do dataset (nomes em inglês das variáveis do Challenge 2019) e
# aparecem tanto na fila quanto no gráfico. Sem a legenda, o painel só é legível para
# quem já as conhece — e o público do painel é a equipe médica, não quem escreveu o
# código. As faixas de referência estão aqui para tornar o gráfico interpretável de
# relance; são valores usuais de adulto, não critério clínico do sistema.
LEGENDA = """
**Legenda**

| Sigla | Significado | Faixa usual (adulto) |
|---|---|---|
| `HR` | frequência cardíaca | 60–100 bpm |
| `SBP` | pressão arterial sistólica | 90–140 mmHg |
| `Resp` | frequência respiratória | 12–20 irpm |
| `O2Sat` | saturação de oxigênio | ≥ 95% |
| `FiO2` | fração inspirada de oxigênio — **dose prescrita** | 0,21 (ar ambiente) a 1,0 |

**Sepse** — disfunção orgânica causada por uma resposta desregulada do organismo a uma
infecção. É o desfecho que serve de *ground-truth* neste trabalho: o dataset marca, hora
a hora, quando o quadro é considerado instalado. O detector **não** usa esse rótulo; ele
aparece apenas na conferência, para medir se o alerta veio antes.

**Colunas da fila** — `Vitais` e `Dose` são contagens de alertas; `Taxa` é a fração da
internação em alerta pelos sinais vitais; `Origem` indica qual série disparou.
"""

COLUNAS = {
    "priority": "Prior.",
    "patient": "Paciente",
    "hours_monitored": "Intern.",
    "vitals_alerts": "Vitais",
    "alert_rate": "Taxa",
    "escalations": "Dose",
    "source": "Origem",
}


def _fila_visivel(fila: pd.DataFrame, prioridades: list[str]) -> pd.DataFrame:
    """Fila filtrada, formatada e renomeada para exibição."""
    if fila.empty:
        return pd.DataFrame(columns=list(COLUNAS.values()))
    sub = fila[fila["priority"].isin(prioridades)] if prioridades else fila.iloc[0:0]
    sub = sub[list(COLUNAS)].copy()
    # a taxa crua (0.7635658914728682) ocupa a coluna inteira e não se lê de relance
    sub["alert_rate"] = (sub["alert_rate"] * 100).round(0).astype(int).astype(str) + "%"
    sub["hours_monitored"] = sub["hours_monitored"].astype(str) + " h"
    return sub.rename(columns=COLUNAS).reset_index(drop=True)


def carregar(limit: int, prioridades: list[str], progress=gr.Progress()):
    """Pontua a coorte, monta a fila e devolve o estado para os cliques seguintes."""
    try:
        detector = vitals.load()
    except FileNotFoundError as e:
        raise gr.Error(str(e)) from e

    progress(0.1, desc=f"Lendo e pontuando {int(limit)} pacientes...")
    serie = alerts.score_cohort(DEFAULT_VITALS_DIR, limit=int(limit), detector=detector)

    progress(0.9, desc="Montando a fila de plantão...")
    fila = alerts.build_queue(serie)

    if fila.empty:
        resumo = "Nenhum alerta ativo na coorte monitorada."
    else:
        c = fila["priority"].value_counts()
        resumo = (f"**{len(fila)} pacientes com alerta** entre os "
                  f"{serie['patient'].nunique()} retidos — "
                  f"ALTA {c.get(alerts.ALTA, 0)} · MEDIA {c.get(alerts.MEDIA, 0)}. "
                  f"Selecione uma linha para ver a série do paciente.")

    return serie, fila, _fila_visivel(fila, prioridades), resumo, None, ""


def filtrar(fila: pd.DataFrame | None, prioridades: list[str]):
    """Refiltra sem repontuar — o estado já tem a fila."""
    if fila is None:
        return pd.DataFrame(columns=list(COLUNAS.values()))
    return _fila_visivel(fila, prioridades)


def abrir_paciente(serie: pd.DataFrame | None, tabela: pd.DataFrame | None,
                   evt: gr.SelectData):
    """Abre o paciente da linha clicada: gráfico + leitura do caso."""
    if serie is None or tabela is None or tabela.empty:
        return None, ""

    linha = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if linha is None or linha >= len(tabela):
        return None, ""
    paciente = str(tabela.iloc[linha]["Paciente"])

    fig = plot_monitor(serie, paciente, str(FIGURES_DIR / f"monitor_{paciente}.png"))

    do_paciente = serie[serie["patient"] == paciente]
    horas_v = do_paciente.loc[do_paciente["is_anomaly"] == 1, "hour"].tolist()
    px = prescriptions.monitor_patient(do_paciente)["summary"]

    L = [f"### Paciente {paciente}", ""]
    L.append(f"**Sinais vitais** — {len(horas_v)} hora(s) em alerta"
             + (f": {', '.join('h' + str(int(h)) for h in horas_v[:15])}"
                + (" ..." if len(horas_v) > 15 else "")
                if horas_v else " (série dentro do padrão aprendido)."))
    L.append("")
    if px["monitored"]:
        L.append(f"**Dose prescrita (FiO2)** — {px['observations']} registros, "
                 f"entre {px['dose_min']:.2f} e {px['dose_max']:.2f}; "
                 + (f"{px['escalations']} escalonamento(s) "
                    f"({', '.join('h' + str(int(h)) for h in px['escalation_hours'])})."
                    if px["escalations"] else "sem escalonamento brusco."))
    else:
        L.append(f"**Dose prescrita (FiO2)** — fora de monitoramento "
                 f"({px['observations']} registros).")
    L.append("")

    # o rótulo aparece só aqui, como conferência, e nunca entra na pontuação
    L.append("**Conferência com o ground-truth** *(não usado na detecção)*")
    if do_paciente["SepsisLabel"].max() == 1:
        onset = int(do_paciente.loc[do_paciente["SepsisLabel"] == 1, "hour"].min())
        na_janela = [h for h in horas_v
                     if onset - vitals.LEAD_WINDOW_HOURS <= h < onset]
        L.append(f"- Sepse registrada a partir da hora {onset}.")
        L.append(f"- Vitais: " + (
            f"alerta {onset - int(min(na_janela))} h antes do início."
            if na_janela else
            f"sem alerta na janela de {vitals.LEAD_WINDOW_HOURS} h que antecede o início."))
        if px.get("lead_hours"):
            L.append(f"- Prescrição: escalonamento {px['lead_hours']} h antes do início.")
    else:
        L.append("- Paciente não desenvolveu sepse.")

    return fig, "\n".join(L)


def carregar_movimentacao(sujeito: int, progress=gr.Progress()):
    """Classifica as leituras de um sujeito e devolve a faixa temporal + a leitura."""
    try:
        detector = movement.load()
    except FileNotFoundError as e:
        raise gr.Error(str(e)) from e

    progress(0.3, desc=f"Classificando as leituras do sujeito {int(sujeito)}...")
    try:
        r = movement.monitor_subject(int(sujeito), DEFAULT_HAR_DIR, detector=detector)
    except ValueError as e:
        raise gr.Error(str(e)) from e

    s, res = r["summary"], r["data"]
    fig = plot_subject_timeline(
        res, int(sujeito), str(FIGURES_DIR / f"timeline_sujeito_{int(sujeito)}.png"))

    por_atividade = (res.groupby("activity")
                        .agg(janelas=("is_anomaly", "size"), alerta=("is_anomaly", "mean"))
                        .sort_values("alerta", ascending=False))

    L = [f"### Sujeito {s['subject']} — {s['windows']} janelas de leitura", ""]
    L.append(f"**{s['alerts']} janelas em alerta** ({s['alert_rate']:.1%} das leituras).")
    L.append("")
    L.append("| Atividade | Classe | Janelas | Em alerta |")
    L.append("|---|---|---:|---:|")
    for atividade, linha in por_atividade.iterrows():
        classe = "repouso" if atividade in movement.REST_ACTIVITIES else "**marcha**"
        L.append(f"| `{atividade}` | {classe} | {int(linha['janelas'])} "
                 f"| {linha['alerta']:.1%} |")
    L.append("")
    if s["recall_movement"] is not None:
        L.append(f"**Recall sobre marcha {s['recall_movement']:.1%}** · "
                 f"falso alarme no repouso {s['false_alarm_rate']:.1%}")
    return fig, "\n".join(L)


def build_demo() -> gr.Blocks:
    """
    Duas abas, uma por fonte de dados.

    A separação não é estética: os pacientes do Challenge 2019 e os sujeitos do UCI HAR
    **não são as mesmas pessoas**. Uma fila única sugeriria que o hospital monitora as
    três séries do mesmo paciente, o que a origem dos dados não sustenta. As abas deixam
    a fronteira explícita, e o texto de cada uma diz de onde vêm os dados.
    """
    with gr.Blocks(title="Painel de Plantão — Entrega 3") as demo:
        gr.Markdown(
            "# 🚨 Painel de Plantão — Alertas Automáticos\n"
            "Fluxo final do alerta à equipe médica, a partir das anomalias detectadas. "
            "Cada aba corresponde a uma fonte de dados; em ambas, os indivíduos "
            "monitorados **não participaram do treino** do respectivo modelo."
        )

        estado_serie = gr.State()
        estado_fila = gr.State()

        with gr.Tab("🛏️ Leitos — sinais vitais e prescrições"):
            gr.Markdown(
                "Pacientes de UTI do **PhysioNet Challenge 2019**. As duas subtarefas "
                "descrevem o mesmo paciente, então compõem uma fila única."
            )
            with gr.Row():
                limite = gr.Slider(200, 5000, value=DEFAULT_LIMIT, step=100,
                                   label="Pacientes na coorte (mais = mais lento)")
                prioridades = gr.CheckboxGroup(
                    [alerts.ALTA, alerts.MEDIA], value=[alerts.ALTA, alerts.MEDIA],
                    label="Prioridades exibidas")
            carregar_btn = gr.Button("▶ Carregar plantão", variant="primary")
            resumo = gr.Markdown()

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**Fila de plantão** — clique numa linha para abrir "
                                "o paciente.")
                    tabela = gr.Dataframe(interactive=False, wrap=True,
                                          label="Alertas ativos")
                    gr.Markdown(LEGENDA)
                with gr.Column(scale=2):
                    gr.Markdown("**Série temporal do paciente** — a faixa laranja é a "
                                "janela de 48 h que antecede o início da sepse; as "
                                "faixas vermelhas marcam as horas em alerta.")
                    grafico = gr.Image(label="Monitoramento", type="filepath")
                    detalhe = gr.Markdown()

            gr.Markdown(
                "---\n"
                "A prioridade vem da confiabilidade **medida**: os sinais vitais têm "
                "AUC 0,555 e servem de **triagem**, não de diagnóstico. Um paciente "
                "sobe para **ALTA** quando duas séries independentes disparam."
            )

        with gr.Tab("🚶 Movimentação do paciente"):
            gr.Markdown(
                "Sujeitos do **UCI HAR**, monitorados por acelerômetro e giroscópio. "
                "**Outro dataset, outras pessoas** — não há correspondência com os "
                "pacientes da aba anterior, e por isso a unidade aqui é o *sujeito*.\n\n"
                "O modelo foi treinado **apenas** em atividades de repouso (`LAYING`, "
                "`SITTING`, `STANDING`), que é o esperado em leito, e nunca viu marcha. "
                "Com AUC 0,9999 é a modalidade de maior confiabilidade medida — a única "
                "adequada a alerta automático."
            )
            with gr.Row():
                sujeito = gr.Dropdown(HAR_TEST_SUBJECTS, value=HAR_TEST_SUBJECTS[0],
                                      label="Sujeito monitorado (retidos para teste)")
                mov_btn = gr.Button("▶ Monitorar sujeito", variant="primary")

            grafico_mov = gr.Image(label="Atividade real e alerta", type="filepath")
            detalhe_mov = gr.Markdown()

            gr.Markdown(
                "---\n"
                "No painel superior, a atividade que o sujeito realizava em cada janela "
                "de leitura; no inferior, se o alerta disparou. Os blocos de marcha "
                "acendem por inteiro e os de repouso ficam quase todos apagados."
            )

        carregar_btn.click(
            carregar, [limite, prioridades],
            [estado_serie, estado_fila, tabela, resumo, grafico, detalhe],
            show_progress_on=[tabela])
        prioridades.change(filtrar, [estado_fila, prioridades], [tabela])
        tabela.select(abrir_paciente, [estado_serie, tabela], [grafico, detalhe])
        mov_btn.click(carregar_movimentacao, [sujeito], [grafico_mov, detalhe_mov],
                      show_progress_on=[grafico_mov])

    return demo


def main() -> None:
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)
    build_demo().launch(server_port=PORT)


if __name__ == "__main__":
    main()
