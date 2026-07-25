# 🏥 Painel Unificado — as três entregas em uma tela

Camada de **apresentação** do Sistema de Monitoramento Hospitalar Multimodal: uma aba por
modalidade, num endereço só.

```bash
python -m src.dashboard.app        # abre em http://localhost:7863
```

| Aba | Entrega | O que faz |
|---|---|---|
| 🎥 Vídeo | 1 | OpenPose sobre a sessão de reabilitação → ângulos, desvios, vídeo anotado |
| 🎙️ Áudio | 2 | Transcribe → Comprehend Medical → Comprehend → relatório bilíngue |
| 🚨 Anomalias e alertas | 3 | fila de plantão (leitos) e monitoramento de movimentação |

## O que este painel não é

**Não é uma fusão dos dados.** As quatro fontes — REHAB24-6, consultas simuladas,
PhysioNet Challenge 2019 e UCI HAR — descrevem **populações distintas, sem nenhum
indivíduo em comum**. Uma tela que somasse as três séries num único "paciente" afirmaria
uma correspondência que os datasets abertos não oferecem.

Daí as duas regras da tela:

1. **Uma aba por modalidade, nunca uma visão combinada.** A aba é a fronteira entre as
   fontes — a mesma razão pela qual a Entrega 3 já usava duas abas internas.
2. **Cada aba declara de onde vêm os dados e qual é a unidade monitorada:** *paciente*
   nos leitos e nas consultas, *sujeito* na movimentação. A unidade muda porque a fonte
   muda.

O rodapé do painel traz essa tabela de fontes, e há um teste que exige que ela continue
lá (`tests/test_dashboard.py`) — a afirmação precisa estar diante de quem usa a tela, não
apenas no relatório técnico.

## Como se relaciona com as apps individuais

O painel **não reimplementa nada**: cada aba chama a `build_ui()` da app da respectiva
entrega, que é a mesma montagem usada pela app autônoma.

```
src/video/app.py    build_ui() ──┐
src/audio/app.py    build_ui() ──┼──► src/dashboard/app.py   (7863, uma aba cada)
src/anomaly/app.py  build_ui() ──┘
        │
        └──► build_demo() ──► app autônoma (7860, 7862, 7861)
```

As três continuam funcionando isoladamente, nas portas de sempre. O painel é uma **porta
de entrada adicional**, não um substituto — as apps individuais seguem documentadas nas
seções 3.11, 4.9 e 5.8 do relatório técnico.

> Se alguma app voltar a montar componentes dentro de `build_demo()` em vez de
> `build_ui()`, a app autônoma continua funcionando e **só o painel** perde aquela aba —
> falha silenciosa que `tests/test_dashboard.py` cobre.

## Requisitos por aba

Cada aba degrada sozinha, sem derrubar as demais:

| Aba | Precisa de |
|---|---|
| Vídeo | binário do OpenPose (`docs/openpose_setup.md`) e os vídeos do REHAB24-6 |
| Áudio | nada, para os casos em cache; credenciais da AWS só para casos novos |
| Anomalias | modelos treinados: `python -m src.anomaly.cli --train --limit 5000` |

## Portas

| Tela | Porta |
|---|---|
| Vídeo (autônoma) | 7860 |
| Anomalias (autônoma) | 7861 |
| Áudio (autônoma) | 7862 |
| **Painel unificado** | **7863** |

As quatro podem ficar abertas ao mesmo tempo.
