# Relatório Técnico — Tech Challenge Fase 4
## Sistema de Monitoramento Hospitalar Multimodal (Vídeo, Áudio e Sinais Vitais)

**Curso:** FIAP AI para DEVs (8IADT)  
**Grupo:** 30  
**Integrantes:**
- Rodrigo de Araújo Rosa
- Elias Maximiano da Silva
- Fábia Gomes de Jesus
- Danilo Pereira

**Data:** Julho de 2026

---

> **Documento vivo.** As seções marcadas com **[Em desenvolvimento]** serão completadas
> conforme as entregas avançam. Este relatório acompanha o estado atual da implementação.

---

## 1. Introdução

Este relatório documenta o desenvolvimento do Tech Challenge — Fase 4, cujo objetivo é
construir um **Sistema de Monitoramento Hospitalar Multimodal** capaz de acompanhar
continuamente pacientes em ambiente hospitalar/UTI a partir de dados de **vídeo, áudio e
séries temporais de sinais vitais**, identificando sinais precoces de risco e emitindo
alertas automáticos para a equipe médica.

A solução integra três entregas técnicas independentes, unidas pelo cenário clínico:
análise de vídeo de sessões de reabilitação (estimação de pose e detecção de desvios
posturais), análise de áudio de consultas (transcrição e biomarcadores vocais) e detecção
de anomalias em sinais vitais, prescrições e padrões de movimentação.

> **AVISO — Uso exclusivamente acadêmico.** Este sistema não deve ser utilizado para
> diagnóstico, prescrição ou qualquer conduta clínica em pacientes reais. Os datasets são
> públicos e, por prática acadêmica padrão, não correspondem aos mesmos pacientes nas três
> modalidades.

### 1.1 Objetivo

Desenvolver um sistema que:

1. Processe vídeos clínicos e detecte movimentos/eventos fora do padrão esperado (análise
   postural com OpenPose), gerando relatórios automáticos de desvios
2. Processe áudios de consultas, transcreva a fala (Azure Speech-to-Text) e identifique
   termos críticos e sentimento (Azure Text Analytics), além de biomarcadores acústicos
3. Aplique técnicas de detecção de anomalias em séries temporais de sinais vitais,
   evolução de prescrições e padrões de movimentação
4. Integre os serviços gerenciados em nuvem (Azure Cognitive Services) e produza um fluxo
   de alerta à equipe médica

### 1.2 Entregáveis do Projeto

| Entregável | Link |
|:---|:---|
| Este Relatório | `reports/TECHNICAL_REPORT_FASE4.md` |
| Repositório | (a publicar) `github.com/rodrigoaraujorosa/fiap-ia-devs-8iadt-fase4-tech-challenge` |
| Vídeo (até 15 min) | (a publicar — YouTube/Vimeo) |
| Notebook — OpenPose no Colab | `notebooks/openpose_rehab24-6_colab.ipynb` |

### 1.3 Visão Geral das Três Entregas

| # | Entrega | Objetivo | Dataset | Modelos / Serviços | Estado |
|:--|:--|:--|:--|:--|:--|
| 1 | Análise de Vídeo | Detectar desvios posturais em vídeos de reabilitação | REHAB24-6 (RGB + rótulos correto/incorreto) | OpenPose (BODY_25), IsolationForest, z-score robusto | Implementada e validada |
| 2 | Análise de Áudio | Detectar alterações vocais/respiratórias | Coswara | Azure Speech-to-Text, Azure Text Analytics, librosa | [Em desenvolvimento] |
| 3 | Detecção de Anomalias | Anomalias em sinais vitais, prescrições e movimentação | PhysioNet Challenge 2019, UCI HAR, Synthea | IsolationForest (baseline) | Baseline implementado |

### 1.4 Datasets Utilizados

| Modalidade | Dataset | Acesso | Observação |
|:--|:--|:--|:--|
| Vídeo | REHAB24-6 | Aberto (Zenodo, ~2,7 GB) | RGB + rótulos de execução correta/incorreta |
| Áudio | Coswara | Open-access (não-comercial) | Respiração, tosse, vogais, dígitos + metadados |
| Sinais vitais | PhysioNet/CinC Challenge 2019 | Aberto (~42 MB) | 40.336 pacientes de UTI, séries horárias |
| Movimentação | UCI HAR | Aberto (~60 MB) | 561 features (acelerômetro + giroscópio), 6 atividades |
| Prescrições | Synthea (sintético) | Aberto | Gerador de registros clínicos sintéticos |

---

## 2. Arquitetura Geral do Sistema

O sistema é composto por três pipelines independentes (um por modalidade) que convergem
para uma camada de fusão e alerta. Cada modalidade produz um sinal de anomalia
interpretável que pode disparar um alerta à equipe médica.

```
[Vídeo: reabilitação]  ── OpenPose ──► keypoints ──► ângulos ──► desvios ─┐
                                                                          │
[Áudio: consulta]  ── Azure Speech ──► transcrição ──► termos/sentimento ─┤
                     └─ librosa ──► biomarcadores acústicos               ├─► fusão / alerta
                                                                          │    à equipe médica
[Sinais vitais / movimentação / prescrições]  ── IsolationForest ────────┘
```

### 2.1 Componentes por Modalidade

| Modalidade | Entrada | Processamento | Saída |
|:--|:--|:--|:--|
| Vídeo | vídeo RGB (.mp4) | OpenPose (BODY_25) → ângulos articulares → detecção de desvios | relatório + vídeo anotado |
| Áudio | áudio de consulta (.wav) | Azure Speech-to-Text + Text Analytics + biomarcadores | transcrição + termos críticos + alterações vocais |
| Anomalias | séries temporais (.psv/.txt/.csv) | IsolationForest + z-score | frames/horas anômalas + alertas |

### 2.2 Princípio de Projeto: Desacoplamento

Cada modalidade é autocontida e independe das demais para executar. A integração ocorre na
camada de alerta, que consome os sinais de anomalia de cada pipeline. Esse desacoplamento
permite desenvolver, testar e avaliar cada entrega separadamente — coerente com a natureza
dos datasets (fontes distintas, sem pacientes compartilhados).

---

## 3. Entrega 1 — Análise de Vídeo (OpenPose)

### 3.1 Visão Geral

O pipeline de vídeo processa gravações de sessões de reabilitação física, estima a pose 2D
do paciente com **OpenPose** e detecta **desvios posturais fora do padrão**, gerando um
relatório automático e um vídeo anotado.

A decisão de arquitetura central: o **OpenPose (binário externo) apenas extrai os keypoints
em JSON; todo o restante do pipeline é Python puro processando esses JSON**. Isso torna o
código independente do ambiente — roda igual na máquina local ou no Google Colab — e isola
a dependência mais pesada (compilação C++/CUDA) em um artefato pré-compilado.

```
vídeo .mp4 ──[OpenPose]──► JSON por frame ──► [pipeline Python] ──► relatório + gráfico + vídeo anotado
```

### 3.2 Dataset: REHAB24-6

O dataset escolhido é o **REHAB24-6**, um conjunto multimodal de exercícios de reabilitação
física publicado abertamente no Zenodo.

| Item | Descrição |
|:--|:--|
| Fonte | [zenodo.org/records/13305826](https://zenodo.org/records/13305826) |
| Acesso | Aberto, download direto (sem credenciamento) |
| Conteúdo | Vídeos RGB (2 câmeras, 30 fps), esqueletos 2D/3D, marcadores, segmentação |
| Tamanho | `videos.zip` ~2,7 GB |
| Exercícios | 6: abdução de braço, arm VW, flexões, abdução de perna, lunge, agachamento |
| Rótulos | Correção binária (correto/incorreto) por repetição + segmentação temporal |
| Repetições | 1.072 (568 corretas, 504 incorretas) |
| Licença | CC BY-NC 4.0 (uso acadêmico) |

**Justificativa da escolha.** O dataset originalmente previsto (KIMORE) tornou-se
indisponível (servidor da instituição fora do ar) e, além disso, os *mirrors* disponíveis
continham apenas dados de esqueleto (sem RGB), o que inviabilizaria o uso do OpenPose. O
REHAB24-6 supera o KIMORE para o objetivo desta entrega: fornece **vídeo RGB** e, crucialmente,
**rótulos explícitos de execução correta/incorreta**, que servem como *ground-truth* para
validar quantitativamente os desvios detectados.

**Rótulos (`Segmentation.csv`).** Cada repetição é descrita por `video_id`,
`repetition_number`, `exercise_id`, intervalo de frames (`first_frame`, `last_frame`),
orientação da câmera e `correctness` (0/1), entre outros campos.

### 3.3 Vídeo de Demonstração: PM_008

Para demonstração e validação foi selecionado o vídeo **PM_008** (exercício 6 — agachamento),
por conter, num mesmo vídeo, execuções corretas e incorretas.

| Atributo | Valor |
|:--|:--|
| Resolução / taxa | 1920x1080, 30 fps |
| Duração | ~2 min 53 s (5.191 frames) |
| Câmera | Camera17 (horizontal) |
| Repetições | 27 agachamentos |
| Correção | 21 corretas, 6 incorretas (repetições 17, 23–27) |

### 3.4 Extração de Pose (OpenPose BODY_25)

A pose é estimada com o modelo **BODY_25** do OpenPose v1.7.0. O binário é invocado sobre o
vídeo com escrita dos keypoints em JSON (um arquivo por frame).

| Parâmetro | Valor | Observação |
|:--|:--|:--|
| Modelo | BODY_25 | 25 juntas 2D com confiança |
| `--net_resolution` | `320x176` (local) / `-1x256` (Colab) | menor = menos VRAM |
| `--render_pose` | 0 | renderização feita pelo nosso overlay |
| Saída | 1 JSON por frame | `pose_keypoints_2d = [x0,y0,c0, ...]` |

**Desempenho e subamostragem.** Na GPU local (NVIDIA MX330, 2 GB), o OpenPose processa a
~1,2 s/frame. Processar os 5.191 frames de PM_008 levaria ~1h45. Para viabilizar a execução
local, o vídeo é **subamostrado para 1 a cada 3 frames** (opção `--frame-step 3` do CLI;
10 fps efetivos, 1.731 frames), reduzindo o tempo do OpenPose para ~35–45 min (varia com a
GPU; o pipeline reporta o tempo medido no output e no relatório) sem perda relevante para o
movimento lento do agachamento. O mapeamento é preservado: o índice `i` do keypoint
corresponde ao frame original `3*i`, permitindo o cruzamento correto com os rótulos (ver 3.8).

### 3.5 Parser de Keypoints (`keypoints.py`)

O parser carrega os JSON do OpenPose em um `DataFrame` (uma linha por frame, colunas
`<Junta>_x`, `<Junta>_y`, `<Junta>_c`), com dois tratamentos importantes:

- **Seleção robusta da pessoa principal:** com múltiplas pessoas na cena (observadores ao
  fundo), escolhe-se a **maior e mais confiante** (área do bounding box × confiança média) —
  o paciente em primeiro plano domina e gente menor ao fundo é ignorada — com **estabilização
  temporal**, favorecendo o candidato mais próximo da seleção do frame anterior para não
  alternar entre pessoas. (A abordagem inicial, "maior confiança total por frame", podia
  travar na pessoa errada e corromper os ângulos.)
- **Descarte de juntas de baixa confiança:** juntas com confiança abaixo de `0.1` (ou não
  detectadas) têm coordenadas marcadas como `NaN`, evitando que ângulos sejam calculados
  sobre pontos espúrios em (0, 0).

### 3.6 Ângulos Posturais (`posture.py`)

As coordenadas 2D são convertidas em uma série temporal de ângulos clinicamente relevantes.
São calculados 8 ângulos articulares de 3 pontos e a inclinação do tronco:

| Ângulo | Juntas (vértice em negrito) |
|:--|:--|
| Cotovelo (D/E) | Ombro — **Cotovelo** — Punho |
| Ombro (D/E) | Cotovelo — **Ombro** — Quadril |
| Quadril (D/E) | Ombro — **Quadril** — Joelho |
| Joelho (D/E) | Quadril — **Joelho** — Tornozelo |
| Inclinação do tronco | vetor Pescoço→MidQuadril em relação à vertical da imagem |

O ângulo em um vértice B é obtido pelo produto escalar dos vetores B→A e B→C:
`ângulo = arccos( (v1·v2) / (|v1||v2|) )`, com propagação de `NaN` quando alguma junta está
ausente.

### 3.7 Detecção de Desvios (`anomaly.py`)

Um frame é sinalizado como desvio pela **união** de duas técnicas complementares:

1. **Z-score robusto por ângulo** (mediana + MAD): sinaliza o frame quando *algum* ângulo se
   afasta muito do seu comportamento típico no vídeo (limiar `|z| > 3.5`). É interpretável —
   identifica-se qual articulação desviou (`worst_angle`).
2. **IsolationForest multivariado** (`contamination = 0.03`): aprende o padrão conjunto dos
   ângulos e marca frames globalmente atípicos (mesma técnica da Entrega 3).

A saída acrescenta, por frame: `z_anomaly`, `iso_anomaly`, `is_anomaly` (união), `worst_angle`
e `anomaly_score` (|z| máximo, severidade).

### 3.8 Validação contra o Ground-Truth (`validate.py`)

A validação cruza os frames sinalizados como desvio com os rótulos `correctness` do
REHAB24-6. Para cada repetição, calcula-se a taxa de frames anômalos; a expectativa é que
**repetições incorretas concentrem mais desvios que as corretas**. O parâmetro `frame_step`
ajusta o mapeamento quando o vídeo foi subamostrado (índice `i` ↔ frame original `N*i`). A
validação é acionada diretamente pelo pipeline com a opção `--segmentation` do CLI, que
imprime a taxa por classe e salva o detalhe por repetição em `reports/validacao_<vídeo>.csv`.

**Resultados quantitativos (PM_008).** O OpenPose foi executado sobre o vídeo subamostrado
(1.731 frames, 10 fps efetivos; extração ~43 min na GPU local, análise ~7 s). A cobertura de
detecção das juntas principais (quadril, joelhos) foi de 100%. O detector sinalizou 576 dos
1.731 frames (33,3%) como desvio. Cruzando com os rótulos `correctness` das 27 repetições
(21 corretas, 6 incorretas):

| Classe da repetição | Taxa média de frames anômalos |
|:--|:--:|
| Correta (21 repetições) | 0,430 |
| Incorreta (6 repetições) | 0,614 |

As repetições **incorretas concentram ~43% mais frames de desvio** que as corretas —
confirmando que o detector separa execução boa de execução ruim. Os eventos de maior
severidade (|z| até 25,7) situam-se em t≈155–171 s, correspondendo exatamente às repetições
23–27 (todas rotuladas como incorretas), com o **joelho direito** (`r_knee`) como ângulo
predominante. A inclinação do tronco atinge 56° nessas repetições, indicando o padrão
clássico de má execução do agachamento (tronco projetado para a frente com flexão de joelho
excessiva).

Estes valores foram **reconfirmados após a adoção da seleção robusta de pessoa** (seção 3.5):
como o PM_008 tem um único paciente em cena, os resultados permanecem idênticos — a nova
seleção só muda o comportamento em vídeos com observadores ao fundo.

> **Observação de método.** A taxa de anomalia é alta mesmo nas repetições corretas (0,430)
> porque o z-score robusto sinaliza os extremos do agachamento (fase de descida) como desvio
> em relação à postura ereta mediana do vídeo. Como esse efeito incide igualmente sobre as
> duas classes, o sinal discriminante é a **diferença relativa** entre corretas e incorretas,
> que é consistente. A repetição 17 (incorreta) teve taxa 0,42, próxima da média das corretas
> — nem toda execução incorreta se separa com a mesma força.

### 3.9 Relatório e Vídeo Anotado (`report.py`, `overlay.py`)

- **Relatório automático (Markdown):** voltado à equipe médica, na ordem **gráfico →
  análise → cobertura → estatística → resumo e eventos**. Quando a validação é usada
  (`--segmentation`), o cabeçalho traz também o **exercício** (ex.: "Agachamento (Ex6)"),
  explicitamente marcado como **rótulo do dataset — não detecção automática** (o modelo
  detecta desvios, não classifica o exercício). A seção *Análise* é gerada
  automaticamente a partir dos dados (articulação mais afetada, concentração temporal dos
  desvios, pico de severidade, inclinação máxima do tronco) em linguagem clínica. A
  estatística dos ângulos traz uma coluna com o nome da articulação em português (Joelho D,
  Ombro E, etc.); os eventos listam intervalos contíguos com articulação predominante e
  severidade. Inclui a ressalva de que não substitui avaliação profissional.
- **Vídeo anotado (`overlay.py`):** desenha o esqueleto BODY_25 sobre o vídeo (OpenCV) e
  **destaca visualmente os frames de desvio** — borda vermelha, o osso do ângulo que desviou
  em vermelho, e o rótulo `DESVIO: <ângulo> (|z|=...)`. Material direto para o vídeo-demo.

### 3.10 Testes

O pipeline é coberto por testes automatizados que usam **keypoints sintéticos** (sem
necessidade de OpenPose nem do dataset): validam o cálculo de ângulos, a detecção de
anomalias injetadas e a renderização do overlay.

---

## 4. Entrega 2 — Análise de Áudio (Azure Cognitive Services)

**[Em desenvolvimento]**

### 4.1 Visão Geral

Processar áudios de consultas médicas para detectar alterações vocais/respiratórias
(cansaço, dificuldades respiratórias, disartria), transcrever a fala e identificar termos
críticos e sentimento.

### 4.2 Dataset: Coswara

Conjunto open-access com 2.635 indivíduos e 9 categorias de som (respiração rápida/lenta,
tosse profunda/rasa, vogais sustentadas, dígitos falados) e metadados de sintomas e
comorbidades — bom encaixe para "dificuldades respiratórias e cansaço".

### 4.3 Pipeline Previsto

1. **Azure Speech-to-Text** — transcrição da fala.
2. **Azure Text Analytics (Azure AI Language)** — sentimento, key phrases e *Text Analytics
   for Health* (extração de entidades médicas: sintomas, diagnósticos, medicações).
3. **Biomarcadores acústicos (librosa)** — jitter, shimmer, F0, MFCC sobre o áudio bruto.

### 4.4 Credenciais

As chaves da Azure são carregadas via `.env` (modelo em `.env.example`) por
`src/common/config.py`, sem versionar segredos.

---

## 5. Entrega 3 — Detecção de Anomalias

### 5.1 Visão Geral

Aplicar detecção de anomalias em três subtarefas: séries temporais de sinais vitais,
padrões de movimentação e evolução de prescrições. O baseline utiliza **IsolationForest**,
treinado de forma não-supervisionada, com os rótulos disponíveis servindo apenas para
validação.

### 5.2 Sinais Vitais — PhysioNet/CinC Challenge 2019

| Item | Descrição |
|:--|:--|
| Dados | 40.336 pacientes de UTI, séries horárias (1 `.psv` por paciente) |
| Variáveis | 8 sinais vitais + 26 laboratoriais + demografia |
| Ground-truth | `SepsisLabel` (0/1) — deterioração clínica |
| Baseline | IsolationForest sobre os 8 sinais vitais (`contamination = 0.02`) |
| Implementação | `src/anomaly/load_challenge2019.py` (testado) |

O loader imputa a mediana nos vitais (muitos `NaN`), treina o IsolationForest e gera, por
paciente, um gráfico dos sinais com as horas sinalizadas como anomalia.

### 5.3 Movimentação do Paciente — UCI HAR

| Item | Descrição |
|:--|:--|
| Dados | Vetores de 561 features (acelerômetro + giroscópio), 6 atividades |
| Enquadramento | Repouso (LAYING/SITTING/STANDING) = normal; movimento inesperado = anomalia |
| Baseline | IsolationForest treinado apenas em atividades de repouso (`contamination = 0.05`) |
| Implementação | `src/anomaly/load_uci_har.py` (testado) |

O modelo é treinado somente nas atividades de repouso e sinaliza como anomalia as amostras
que fogem desse padrão (ex.: movimento intenso, análogo a uma queda).

### 5.4 Evolução de Prescrições — Synthea

Sem fonte pública aberta granular de prescrições, adota-se o **Synthea** (gerador sintético)
para produzir séries temporais de medicações por paciente (`START`, `STOP`, `PATIENT`,
`CODE`, `DESCRIPTION`), documentando na banca o uso de dados sintéticos por indisponibilidade
de fonte aberta. **[Em desenvolvimento]**

### 5.5 Geração de Alertas

As horas/amostras sinalizadas como anomalia alimentam a camada de alerta (Seção 6). **[Em
desenvolvimento]**

---

## 6. Integração em Nuvem (Azure) e Fluxo de Alerta

**[Em desenvolvimento]**

O enunciado exige integração com serviços gerenciados em nuvem (Azure Cognitive Services).
O mapeamento realista por modalidade:

| Modalidade | Serviço Azure | Papel | Estado |
|:--|:--|:--|:--|
| Áudio | Azure AI Speech (Speech-to-Text) | Transcrição | Obrigatório |
| Áudio | Azure AI Language (Text Analytics / Health) | Termos críticos + sentimento | Obrigatório |
| Vídeo | — (OpenPose local) | Azure não faz pose clínica | Local |
| Anomalias | — (IsolationForest local) | Ver nota abaixo | Local |
| Fusão/alerta | Azure Functions / Communication Services (ou Logic Apps) | Orquestração + alerta à equipe | A definir |

**Nota sobre o Azure Anomaly Detector.** O serviço que seria o encaixe natural para
séries temporais está em processo de aposentadoria (retirada em 1º de outubro de 2026) e
**não permite criar novos recursos desde 20 de setembro de 2023**. Portanto, a Entrega 3
permanece com detecção local (IsolationForest), decisão documentada na banca.

---

## 7. Decisões de Projeto e Justificativas

### 7.1 Vídeo: OpenPose como Extrator Externo (JSON), Análise em Python

Em vez de usar a API Python do OpenPose (que exige compilação em C++/CUDA), o binário é
invocado por linha de comando e grava keypoints em JSON. Todo o restante é Python puro. Isso
isola a dependência pesada, torna o pipeline portátil (local ou Colab) e simplifica os
testes (keypoints sintéticos).

### 7.2 Troca de Dataset: KIMORE → REHAB24-6

O KIMORE ficou indisponível (servidor fora do ar) e os *mirrors* traziam apenas esqueleto
(sem RGB), inviabilizando o OpenPose. O REHAB24-6 fornece RGB e rótulos de correção — melhor
para o objetivo — e é aberto no Zenodo.

### 7.3 Subamostragem de Frames para Viabilizar a GPU Local

A GPU local (MX330, 2 GB) processa a ~1,2 s/frame. Subamostrar 1 a cada 3 frames reduz o
tempo em ~3x, adequado ao movimento lento do agachamento, preservando o mapeamento de frames
para validação.

### 7.4 Detecção de Anomalias Local em vez do Azure Anomaly Detector

O Azure Anomaly Detector não pode ser provisionado (aposentadoria/retirada). Mantém-se o
IsolationForest local, tecnicamente adequado e reprodutível.

### 7.5 Baseline IsolationForest Comum às Entregas 1 e 3

O IsolationForest é reutilizado como detector não-supervisionado tanto sobre os ângulos
posturais (vídeo) quanto sobre os sinais vitais e a movimentação (anomalias), padronizando a
abordagem e facilitando a comparação.

---

## 8. Estrutura do Repositório

```
fiap-ia-devs-8iadt-fase4-tech-challenge/
├── src/
│   ├── common/                 # config e utilitários compartilhados (Azure, caminhos)
│   ├── video/                  # Entrega 1 — análise de vídeo (OpenPose)
│   │   ├── keypoints.py        # parser dos JSON BODY_25
│   │   ├── posture.py          # ângulos articulares por frame
│   │   ├── anomaly.py          # detecção de desvios (z-score + IsolationForest)
│   │   ├── report.py           # relatório Markdown + gráfico
│   │   ├── overlay.py          # vídeo anotado (esqueleto + desvios)
│   │   ├── validate.py         # validação contra os rótulos do REHAB24-6
│   │   ├── run_openpose.py     # invocação do binário OpenPose
│   │   └── cli.py              # pipeline fim-a-fim
│   ├── audio/                  # Entrega 2 — análise de áudio (Azure)  [Em desenvolvimento]
│   └── anomaly/                # Entrega 3 — detecção de anomalias
│       ├── load_challenge2019.py   # loader + baseline (sinais vitais)
│       └── load_uci_har.py         # loader + baseline (movimentação)
├── data/                       # datasets baixados localmente (não versionado)
├── docs/                       # enunciado + guias (datasets, setup do OpenPose)
├── notebooks/                  # openpose_rehab24-6_colab.ipynb
├── reports/                    # relatório técnico e figuras
├── tests/                      # testes automatizados
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 9. Tecnologias Utilizadas

### 9.1 Infraestrutura

| Tecnologia | Uso |
|:--|:--|
| Python 3.10+ | Linguagem principal |
| OpenPose v1.7.0 (BODY_25) | Estimação de pose 2D (binário externo) |
| GPU local NVIDIA MX330 (2 GB) | Execução local do OpenPose |
| Google Colab (GPU T4) | Execução do OpenPose em lote (alternativa) |
| Azure Cognitive Services | Speech-to-Text e Text Analytics (Entrega 2) |

### 9.2 Bibliotecas Python

| Biblioteca | Uso |
|:--|:--|
| `numpy`, `pandas` | Manipulação de dados e séries |
| `scikit-learn` | IsolationForest, imputação |
| `matplotlib`, `seaborn` | Gráficos e relatórios |
| `opencv-python` | Leitura de vídeo e overlay do esqueleto |
| `azure-cognitiveservices-speech` | Azure Speech-to-Text (Entrega 2) |
| `azure-ai-textanalytics` | Azure Text Analytics (Entrega 2) |
| `librosa`, `soundfile` | Biomarcadores acústicos (Entrega 2) |
| `python-dotenv` | Carregamento de credenciais Azure |
| `pytest` | Testes automatizados |

---

## 10. Reprodutibilidade

### 10.1 Pré-requisitos

- Python 3.10+ e as dependências de `requirements.txt`
- OpenPose v1.7.0 (binário portátil) — ver `docs/openpose_setup.md`
- (Entrega 2) Conta Azure com recursos de Speech e Language; credenciais em `.env`

### 10.2 Instalação

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows (Git Bash)
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

### 10.3 Execução — Entrega 1 (Análise de Vídeo)

```bash
# Vídeo -> OpenPose -> relatório + vídeo anotado + validação, num comando:
#   --frame-step 3  subamostra (GPU fraca)
#   --overlay       gera o vídeo anotado
#   --segmentation  valida contra os rótulos correto/incorreto
python -m src.video.cli --video data/video/rehab24-6/PM_008-Camera17-30fps.mp4 \
    --openpose-root tools/openpose --fps 30 --frame-step 3 --overlay \
    --segmentation data/video/rehab24-6/Segmentation.csv

# A partir de JSONs já extraídos (ex.: gerados no Colab, sem subamostragem)
python -m src.video.cli --json-dir reports/json/PM_008 --fps 30
```

Sem GPU adequada, usar o notebook `notebooks/openpose_rehab24-6_colab.ipynb` (GPU gratuita do
Colab) para extrair os JSONs e rodar a análise em seguida.

### 10.4 Execução — Entrega 3 (Detecção de Anomalias)

```bash
python src/anomaly/load_challenge2019.py --data ./data/anomaly/challenge2019 --patient p000001
python src/anomaly/load_uci_har.py --data "./data/anomaly/uci_har/UCI HAR Dataset"
```

### 10.5 Testes

```bash
pytest -q
```

---

## 11. Limitações e Trabalhos Futuros

### 11.1 Limitações Atuais

| Limitação | Impacto | Mitigação Possível |
|:--|:--|:--|
| GPU local fraca (MX330, 2 GB) | OpenPose lento (~1,2 s/frame) | Subamostragem; execução no Colab (T4) |
| Datasets de modalidades distintas | Não há paciente comum entre vídeo, áudio e vitais | Prática acadêmica padrão; documentado |
| Azure Anomaly Detector aposentado | Sem serviço gerenciado para séries temporais | Detecção local (IsolationForest) |
| Entrega 2 (áudio) em desenvolvimento | Modalidade ainda não integrada | Concluir pipeline Azure + Coswara |
| Detecção não-supervisionada | Limiares definidos empiricamente | Calibração com os rótulos disponíveis |

### 11.2 Trabalhos Futuros

1. Concluir a Entrega 2 (Azure Speech-to-Text + Text Analytics + biomarcadores)
2. Completar a validação quantitativa do vídeo contra os rótulos do REHAB24-6
3. Implementar a camada de fusão e o fluxo de alerta em nuvem
4. Calibrar os limiares de anomalia com os *ground-truths* disponíveis
5. Processar mais vídeos/exercícios do REHAB24-6 para robustez
6. Avaliar modelos temporais (LSTM/autoencoder) para as séries de sinais vitais

---

## 12. Conclusão

**[Em desenvolvimento — a consolidar ao final das três entregas.]**

Até o momento, a Entrega 1 (Análise de Vídeo) está implementada e validada: o pipeline
OpenPose → ângulos posturais → detecção de desvios → relatório e vídeo anotado foi executado
sobre o vídeo PM_008 do REHAB24-6 (agachamentos), e a validação contra os rótulos de execução
mostrou taxa média de desvio de **0,614 nas repetições incorretas contra 0,430 nas corretas**
— separação consistente na direção esperada, com os desvios mais severos coincidindo com as
repetições rotuladas como incorretas. A Entrega 3 (Detecção de Anomalias) possui baseline
funcional (IsolationForest) para sinais vitais e movimentação. A Entrega 2 (Análise de Áudio)
e a camada de integração em nuvem estão em desenvolvimento.

---

## Referências

1. REHAB24-6: A multi-modal dataset of physical rehabilitation exercises. Zenodo. [zenodo.org/records/13305826](https://zenodo.org/records/13305826)

2. Cao, Z., Hidalgo, G., Simon, T., Wei, S.-E., & Sheikh, Y. (2019). OpenPose: Realtime Multi-Person 2D Pose Estimation Using Part Affinity Fields. *IEEE TPAMI*.

3. Reyna, M. A., et al. (2019). Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019. [physionet.org/content/challenge-2019](https://physionet.org/content/challenge-2019/1.0.0/)

4. Anguita, D., et al. (2013). A Public Domain Dataset for Human Activity Recognition Using Smartphones (UCI HAR). *ESANN 2013*.

5. Sharma, N., et al. (2020). Coswara — A Database of Breathing, Cough, and Voice Sounds for COVID-19 Diagnosis. [github.com/iiscleap/Coswara-Data](https://github.com/iiscleap/Coswara-Data)

6. Microsoft Azure. Cognitive Services / AI Services — Speech and Language documentation. [learn.microsoft.com/azure/ai-services](https://learn.microsoft.com/azure/ai-services)
