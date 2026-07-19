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

from . import alerts, prescriptions, vitals
from .report import FIGURES_DIR, plot_monitor

DEFAULT_VITALS_DIR = "data/anomaly/challenge2019"
DEFAULT_LIMIT = 1000

# A porta 7860 é do app da Entrega 1; os dois podem ficar abertos lado a lado na demo.
PORT = 7861

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


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Painel de Plantão — Entrega 3") as demo:
        gr.Markdown(
            "# 🚨 Painel de Plantão — Alertas Automáticos\n"
            "Fluxo final do alerta à equipe médica. A fila é montada a partir das "
            "anomalias detectadas em **sinais vitais** e **evolução da dose prescrita**, "
            "sobre pacientes que o modelo não viu no treino."
        )

        estado_serie = gr.State()
        estado_fila = gr.State()

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
                gr.Markdown("**Fila de plantão** — clique numa linha para abrir o paciente.")
                tabela = gr.Dataframe(interactive=False, wrap=True,
                                      label="Alertas ativos")
            with gr.Column(scale=2):
                gr.Markdown("**Série temporal do paciente** — a faixa laranja é a janela "
                            "de 48 h que antecede o início da sepse; as faixas "
                            "vermelhas marcam as horas em alerta.")
                grafico = gr.Image(label="Monitoramento", type="filepath")
                detalhe = gr.Markdown()

        gr.Markdown(
            "---\n"
            "A prioridade vem da confiabilidade **medida** de cada modalidade: "
            "movimentação AUC 0,9999 (alerta automático), sinais vitais AUC 0,555 "
            "(triagem). Um paciente sobe para **ALTA** quando duas séries independentes "
            "disparam. **Sinais vitais não constituem diagnóstico** — esta fila é de "
            "triagem, para revisão humana."
        )

        carregar_btn.click(
            carregar, [limite, prioridades],
            [estado_serie, estado_fila, tabela, resumo, grafico, detalhe],
            show_progress_on=[tabela])
        prioridades.change(filtrar, [estado_fila, prioridades], [tabela])
        tabela.select(abrir_paciente, [estado_serie, tabela], [grafico, detalhe])

    return demo


def main() -> None:
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)
    build_demo().launch(server_port=PORT)


if __name__ == "__main__":
    main()
