"""
Tela do Sistema de Monitoramento Hospitalar Multimodal.

É a **interface do sistema para a equipe médica**: o ponto único por onde o profissional
acompanha o paciente internado, com uma aba por modalidade de monitoramento — plantão e
alertas, sessão de reabilitação em vídeo e consulta em áudio.

A tela abre na aba de **alertas**, e não na primeira entrega. Quem abre um sistema de
monitoramento em plantão quer saber, antes de tudo, se há paciente precisando de atenção;
as demais abas são consultadas a partir de uma pergunta, o alerta chega sem ser pedido. A
ordem das abas continua sendo a das entregas, para preservar a correspondência com o
enunciado e com o relatório técnico.

**O que esta tela NÃO faz: fundir os dados das três modalidades.** REHAB24-6, consultas
simuladas, Challenge 2019 e UCI HAR descrevem **populações distintas**, sem nenhum
indivíduo em comum (1.4 e 2.2 do relatório técnico). A consequência aqui é clínica, não
estética: uma tela que somasse as três séries num único "paciente" levaria a equipe a ler
como um histórico o que são pessoas diferentes. Daí as duas regras:

1. **Uma aba por modalidade, nunca uma visão combinada.** A aba é a fronteira entre as
   fontes, como já eram as duas abas internas da Entrega 3 (5.8).
2. **Cada aba declara de onde vêm os dados e qual é a unidade monitorada** — *paciente*
   nos leitos e nas consultas, *sujeito* na movimentação. A unidade muda porque a fonte
   muda.

As abas **não reimplementam nada**: cada uma chama a ``build_ui()`` da app da respectiva
entrega. Essas apps individuais continuam existindo, nas portas 7860, 7862 e 7861, como
**pontos de entrada por entrega** — úteis para desenvolver e para demonstrar uma
modalidade isolada. Para a equipe médica, porém, o sistema é esta tela.

Uso:
    python -m src.dashboard.app        # abre em http://localhost:7863

Requisitos por aba (cada uma degrada sozinha, sem derrubar as demais):
    Alertas    modelos treinados (python -m src.anomaly.cli --train --limit 5000)
    Vídeo      binário do OpenPose e os vídeos do REHAB24-6 em data/video/
    Áudio      nada, para os casos em cache; credenciais da AWS para casos novos
"""
from __future__ import annotations

from pathlib import Path

import gradio as gr

from ..anomaly.app import build_ui as build_anomaly_ui
from ..anomaly.report import FIGURES_DIR
from ..audio.app import build_ui as build_audio_ui
from ..video.app import build_ui as build_video_ui

# 7860, 7861 e 7862 são das apps por entrega — a tela do sistema não toma a porta de
# nenhuma delas, para que todas possam ficar abertas ao mesmo tempo.
PORT = 7863

# A aba que o sistema mostra ao abrir. Ver a justificativa no docstring do módulo.
ABA_INICIAL = "alertas"

CABECALHO = """
# 🏥 Monitoramento Hospitalar Multimodal

Acompanhamento do paciente internado. Cada aba é uma modalidade de monitoramento:
**alertas** de sinais vitais, dose prescrita e movimentação; **vídeo** de sessões de
reabilitação; e **áudio** de consultas.
"""

RODAPE = """
---
### Origem dos dados monitorados

Cada modalidade vem de um **dataset público distinto, e não há indivíduo em comum entre
eles** — é a prática usual quando cada modalidade tem sua própria fonte aberta.

| Aba | Dataset | Unidade monitorada |
|:--|:--|:--|
| Alertas — leitos | PhysioNet/CinC Challenge 2019 | paciente de UTI |
| Alertas — movimentação | UCI HAR | sujeito |
| Vídeo | REHAB24-6 (reabilitação física) | paciente em sessão |
| Áudio | Consultas médicas simuladas (figshare) | paciente em consulta |

Por isso as abas **não se combinam** num histórico único: o que aparece numa aba não
descreve a mesma pessoa que aparece na outra. Cada aba é autocontida, e o que as une é a
arquitetura comum — mesma estrutura de módulos, mesma disciplina de validação contra
*ground-truth* e o mesmo baseline não supervisionado onde ele cabe.

> **Uso exclusivamente acadêmico.** Nenhuma saída desta tela constitui diagnóstico ou
> conduta clínica. Os alertas são **triagem para revisão humana**: os sinais vitais têm
> AUC 0,555 e servem para ordenar a atenção, não para decidir.
"""


def build_demo() -> gr.Blocks:
    """Monta a tela: cabeçalho, uma aba por modalidade e o rodapé sobre as fontes."""
    with gr.Blocks(title="Monitoramento Hospitalar Multimodal") as demo:
        gr.Markdown(CABECALHO)

        # A ordem das abas é a das entregas (vídeo, áudio, anomalias), mas a aba aberta
        # por padrão é a de alertas — é o que a equipe procura primeiro.
        with gr.Tabs(selected=ABA_INICIAL):
            with gr.Tab("🎥 Vídeo", id="video"):
                build_video_ui()
            with gr.Tab("🎙️ Áudio", id="audio"):
                build_audio_ui()
            with gr.Tab("🚨 Anomalias e alertas", id=ABA_INICIAL):
                build_anomaly_ui()

        gr.Markdown(RODAPE)
    return demo


def main() -> None:
    # A aba de alertas grava as figuras do monitoramento em disco; sem o diretório, o
    # primeiro clique falharia. Mesma preparação que a app da Entrega 3 faz.
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)
    build_demo().launch(server_port=PORT)


if __name__ == "__main__":
    main()
