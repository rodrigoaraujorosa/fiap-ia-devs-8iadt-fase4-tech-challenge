"""
Painel unificado do Sistema de Monitoramento Hospitalar Multimodal.

Reúne as três entregas numa única tela, uma aba por modalidade — vídeo, áudio e
detecção de anomalias. É a camada de **apresentação** do sistema: quem demonstra o
trabalho abre um endereço só, em vez de três.

**O que este painel NÃO é: uma fusão dos dados.** As três modalidades continuam
independentes, e a razão está nos datasets: REHAB24-6, consultas simuladas, Challenge
2019 e UCI HAR descrevem **populações distintas**, sem nenhum indivíduo em comum (ver
1.4 e 2.2 do relatório técnico). Uma tela que somasse as três séries num único
"paciente" afirmaria uma correspondência que as fontes abertas não oferecem.

Daí as duas regras de projeto desta tela:

1. **Uma aba por modalidade, nunca uma visão combinada.** A aba é a fronteira entre as
   fontes, como já eram as duas abas internas da Entrega 3 (5.8).
2. **Cada aba declara de onde vêm os dados e qual é a unidade monitorada** — *paciente*
   nos leitos e nas consultas, *sujeito* na movimentação. A unidade muda porque a fonte
   muda.

As abas **não reimplementam nada**: cada uma chama a ``build_ui()`` da app da respectiva
entrega, que é a mesma montagem usada pelas apps autônomas. As três continuam
funcionando isoladamente, nas portas de sempre (7860, 7862 e 7861) — este painel é uma
porta de entrada adicional, não um substituto.

Uso:
    python -m src.dashboard.app        # abre em http://localhost:7863

Requisitos por aba (cada uma degrada sozinha, sem derrubar as demais):
    Vídeo      binário do OpenPose e os vídeos do REHAB24-6 em data/video/
    Áudio      nada, para os casos em cache; credenciais da AWS para casos novos
    Anomalias  modelos treinados (python -m src.anomaly.cli --train --limit 5000)
"""
from __future__ import annotations

from pathlib import Path

import gradio as gr

from ..anomaly.app import build_ui as build_anomaly_ui
from ..anomaly.report import FIGURES_DIR
from ..audio.app import build_ui as build_audio_ui
from ..video.app import build_ui as build_video_ui

# 7860, 7861 e 7862 são das apps autônomas — o painel não toma a porta de nenhuma
# delas, para que as quatro possam ficar abertas ao mesmo tempo na demonstração.
PORT = 7863

CABECALHO = """
# 🏥 Sistema de Monitoramento Hospitalar Multimodal

Três modalidades de monitoramento do paciente internado, uma aba para cada:
**vídeo** de sessões de reabilitação, **áudio** de consultas e **detecção de anomalias**
em séries temporais.
"""

RODAPE = """
---
### Sobre as fontes de dados

As três modalidades usam **datasets públicos distintos e não há indivíduo em comum entre
eles** — é a prática usual quando cada modalidade tem sua própria fonte aberta.

| Aba | Dataset | Unidade monitorada |
|:--|:--|:--|
| Vídeo | REHAB24-6 (reabilitação física) | paciente em sessão |
| Áudio | Consultas médicas simuladas (figshare) | paciente em consulta |
| Anomalias — leitos | PhysioNet/CinC Challenge 2019 | paciente de UTI |
| Anomalias — movimentação | UCI HAR | sujeito |

Por isso este painel **não combina** as três num único paciente: cada aba é
autocontida, e o que as une é a arquitetura comum — mesma estrutura de módulos, mesma
disciplina de validação contra *ground-truth* e o mesmo baseline não supervisionado onde
ele cabe.

> **Uso exclusivamente acadêmico.** Nenhuma das saídas constitui diagnóstico ou conduta
> clínica. Os alertas da aba de anomalias são **triagem para revisão humana**.
"""


def build_demo() -> gr.Blocks:
    """Monta o painel: cabeçalho, uma aba por modalidade e o rodapé sobre as fontes."""
    with gr.Blocks(title="Monitoramento Hospitalar Multimodal") as demo:
        gr.Markdown(CABECALHO)

        with gr.Tabs():
            # A ordem é a das entregas no enunciado e no relatório técnico.
            with gr.Tab("🎥 Vídeo"):
                build_video_ui()
            with gr.Tab("🎙️ Áudio"):
                build_audio_ui()
            with gr.Tab("🚨 Anomalias e alertas"):
                build_anomaly_ui()

        gr.Markdown(RODAPE)
    return demo


def main() -> None:
    # A aba de anomalias grava as figuras do monitoramento em disco; sem o diretório,
    # o primeiro clique falharia. Mesma preparação que a app autônoma faz.
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)
    build_demo().launch(server_port=PORT)


if __name__ == "__main__":
    main()
