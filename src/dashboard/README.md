# 🏥 Tela do Sistema — Monitoramento Hospitalar Multimodal

Interface do sistema **para a equipe médica**: o ponto único por onde o profissional
acompanha o paciente internado.

```bash
python -m src.dashboard.app        # abre em http://localhost:7863
```

| Aba | Entrega | O que a equipe faz aqui |
|---|---|---|
| 🚨 Anomalias e alertas | 3 | vê a fila de plantão e abre a série do paciente que alertou |
| 🎥 Vídeo | 1 | processa a sessão de reabilitação e vê os desvios posturais |
| 🎙️ Áudio | 2 | ouve a consulta e lê os achados clínicos extraídos da fala |

## Abre no plantão, não na primeira entrega

Quem abre um sistema de monitoramento quer saber, antes de tudo, **se há paciente
precisando de atenção**. As outras duas abas são consultadas a partir de uma pergunta que
o profissional já tem; o alerta chega sem ser pedido.

A **ordem** das abas continua sendo a das entregas — para preservar a correspondência com
o enunciado e com o relatório —, mas a aba **aberta por padrão** é a de alertas. Há um
teste que fixa as duas coisas ao mesmo tempo (`tests/test_dashboard.py`).

## O que esta tela não faz

**Não funde os dados das três modalidades.** As quatro fontes — REHAB24-6, consultas
simuladas, PhysioNet Challenge 2019 e UCI HAR — descrevem **populações distintas, sem
nenhum indivíduo em comum**.

A consequência aqui é clínica, não estética: uma tela que somasse as três séries num
único "paciente" levaria a equipe a ler como um histórico o que são pessoas diferentes.
Daí as duas regras:

1. **Uma aba por modalidade, nunca uma visão combinada.** A aba é a fronteira entre as
   fontes — a mesma razão pela qual a Entrega 3 já usava duas abas internas.
2. **Cada aba declara de onde vêm os dados e qual é a unidade monitorada:** *paciente*
   nos leitos e nas consultas, *sujeito* na movimentação. A unidade muda porque a fonte
   muda.

O rodapé traz essa tabela de fontes e a ressalva de que os alertas são **triagem para
revisão humana**, não decisão. Testes exigem que as duas coisas continuem lá — precisam
estar diante de quem usa a tela, não apenas no relatório técnico.

## Relação com as apps por entrega

Esta tela **não reimplementa nada**: cada aba chama a `build_ui()` da app da respectiva
entrega.

```
src/video/app.py    build_ui() ──┐
src/audio/app.py    build_ui() ──┼──► src/dashboard/app.py   (7863, tela do sistema)
src/anomaly/app.py  build_ui() ──┘
        │
        └──► build_demo() ──► app por entrega (7860, 7862, 7861)
```

As apps individuais continuam existindo como **pontos de entrada por entrega** — úteis
para desenvolver e para demonstrar uma modalidade isolada, e é a elas que se referem as
seções 3.11, 4.9 e 5.8 do relatório. Para a equipe médica, porém, o sistema é esta tela.

> Se alguma app voltar a montar componentes dentro de `build_demo()` em vez de
> `build_ui()`, ela continua funcionando sozinha e **só esta tela** perde aquela aba —
> falha silenciosa que `tests/test_dashboard.py` cobre.

## Requisitos por aba

Cada aba degrada sozinha, sem derrubar as demais:

| Aba | Precisa de |
|---|---|
| Anomalias e alertas | modelos treinados: `python -m src.anomaly.cli --train --limit 5000` |
| Vídeo | binário do OpenPose (`docs/openpose_setup.md`) e os vídeos do REHAB24-6 |
| Áudio | nada, para os casos em cache; credenciais da AWS só para casos novos |

## Portas

| Tela | Porta |
|---|---|
| **Sistema (esta tela)** | **7863** |
| Vídeo — app da Entrega 1 | 7860 |
| Anomalias — app da Entrega 3 | 7861 |
| Áudio — app da Entrega 2 | 7862 |

As quatro podem ficar abertas ao mesmo tempo.
