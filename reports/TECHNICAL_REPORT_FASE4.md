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
2. Processe áudios de consultas, transcreva a fala (Amazon Transcribe) e identifique
   termos críticos (Amazon Comprehend Medical), além de biomarcadores acústicos
3. Aplique técnicas de detecção de anomalias em séries temporais de sinais vitais,
   evolução de prescrições e padrões de movimentação
4. Integre serviços gerenciados em nuvem (AWS — ver 4.2 sobre a troca de provedor) e
   produza um fluxo de alerta à equipe médica

### 1.2 Entregáveis do Projeto

| Entregável | Link |
|:---|:---|
| Este Relatório | `reports/TECHNICAL_REPORT_FASE4.md` |
| Repositório | https://github.com/rodrigoaraujorosa/fiap-ia-devs-8iadt-fase4-tech-challenge |
| Vídeo (até 15 min) | (a publicar — YouTube/Vimeo) |
| App de demonstração (Gradio) | `python -m src.video.app` |

### 1.3 Visão Geral das Três Entregas

| # | Entrega | Objetivo | Dataset | Modelos / Serviços | Estado |
|:--|:--|:--|:--|:--|:--|
| 1 | Análise de Vídeo | Detectar desvios posturais em vídeos de reabilitação | REHAB24-6 (RGB + rótulos correto/incorreto) | OpenPose (BODY_25), IsolationForest, z-score robusto | Implementada e validada |
| 2 | Análise de Áudio | Transcrever a fala, extrair achados clínicos e medir biomarcadores | Consultas simuladas + Coswara | Amazon Transcribe, Comprehend Medical, Translate; Praat/librosa | Implementada |
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
[Áudio: consulta]  ── Transcribe ──► transcrição ──► Comprehend Medical ─┤
                     └─ Praat/librosa ──► biomarcadores acústicos         ├─► fusão / alerta
                                                                          │    à equipe médica
[Sinais vitais / movimentação / prescrições]  ── IsolationForest ────────┘
```

### 2.1 Componentes por Modalidade

| Modalidade | Entrada | Processamento | Saída |
|:--|:--|:--|:--|
| Vídeo | vídeo RGB (.mp4) | OpenPose (BODY_25) → ângulos articulares → detecção de desvios | relatório + vídeo anotado |
| Áudio | consulta (.mp3) e fonação (.wav) | Amazon Transcribe + Comprehend Medical + biomarcadores (Praat) | transcrição + achados clínicos + medidas vocais |
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
código independente do ambiente (portátil, roda em qualquer máquina) — e isola
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

### 3.3 Vídeos Utilizados: Três Experimentos

Foram executados **três experimentos** sobre vídeos do REHAB24-6, todos contendo execuções
corretas e incorretas no mesmo vídeo. Não são três repetições do mesmo teste: cada um isola
uma variável diferente.

| Atributo | PM_008 | PM_034 | PM_006 |
|:--|:--|:--|:--|
| Papel | validação quantitativa | demonstração | condição de captura adversa |
| Exercício | Ex6 — agachamento | Ex4 — abdução de perna | Ex4 — abdução de perna |
| Subtipo | — | perna esquerda | perna direita |
| Sujeito (`person_id`) | 1 | 4 | 1 |
| Resolução / taxa | 1920x1080, 30 fps | 1920x1080, 30 fps | 1920x1080, 30 fps |
| Duração | ~2 min 53 s (5.191 frames) | ~37 s (1.108 frames) | ~35 s (1.042 frames) |
| Orientação da câmera | frontal e meio-perfil | frontal | meio-perfil |
| Iluminação (`lights_on`) | 0 | 1 | 0 |
| Repetições | 27 | 10 | 10 |
| Correção | 21 corretas, 6 incorretas | 5 corretas, 5 incorretas | 5 corretas, 5 incorretas |
| `extra_person_in_cam17` | 0 | 0 | 3 |
| Pessoas extras observadas | 0 | 0 | **1** |

Os papéis:

- **PM_008** é o caso mais extenso (27 repetições) e o de maior volume de frames — a
  validação quantitativa principal, e o único de exercício diferente.
- **PM_034** é a execução em **condições favoráveis**: câmera frontal, luzes acesas, cena
  sem outras pessoas. É o vídeo da demonstração.
- **PM_006** é a execução em **condições adversas**: câmera em meio-perfil, luzes apagadas
  e uma pessoa ao fundo. Comparado ao PM_034, mede quanto o pipeline degrada fora do
  cenário ideal (resultados em 3.8).

> **Este par não é um experimento controlado.** PM_034 e PM_006 compartilham o exercício e a
> estrutura de repetições (10, sendo 5 corretas e 5 incorretas), mas diferem em **pelo menos
> cinco variáveis simultâneas**: sujeito (`person_id` 4 e 1), perna trabalhada (esquerda e
> direita), orientação da câmera (frontal e meio-perfil), iluminação (acesa e apagada) e
> presença de uma pessoa ao fundo. A diferença de desempenho entre eles é real e medida, mas
> **não pode ser atribuída a nenhuma dessas variáveis isoladamente** — em particular, a
> iluminação apagada é uma explicação tão plausível quanto o observador para a perda de
> qualidade na detecção de juntas. Isolar cada fator exigiria selecionar pares de vídeos que
> variem uma variável por vez, o que o REHAB24-6 permite e fica registrado como trabalho
> futuro.

> **Nota sobre o rótulo `extra_person_in_cam17`.** O campo assume os valores 0, 1, 2 e 3 no
> `Segmentation.csv`, mas a documentação do REHAB24-6 não acompanha o arquivo e **não foi
> possível confirmar se o valor é uma contagem de pessoas ou um código de categoria**. A
> inspeção direta do PM_006 (valor 3) mostra **uma única pessoa extra** em cena, o que
> descarta a leitura "valor = número de pessoas". A contagem usada neste relatório é
> portanto **empírica, medida sobre os próprios keypoints**: o OpenPose detecta 2 pessoas em
> 302 dos 348 frames do PM_006 (87%) e 1 pessoa em 361 dos 370 frames do PM_034 (97,6%).
> Para o argumento desta seção basta a diferença entre as cenas, que é inequívoca; o
> significado exato do rótulo fica em aberto.

### 3.4 Extração de Pose (OpenPose BODY_25)

A pose é estimada com o modelo **BODY_25** do OpenPose v1.7.0. O binário é invocado sobre o
vídeo com escrita dos keypoints em JSON (um arquivo por frame).

| Parâmetro | Valor | Observação |
|:--|:--|:--|
| Modelo | BODY_25 | 25 juntas 2D com confiança |
| `--net_resolution` | `320x176` (padrão local) | menor = menos VRAM |
| `--render_pose` | 0 | renderização feita pelo nosso overlay |
| Saída | 1 JSON por frame | `pose_keypoints_2d = [x0,y0,c0, ...]` |

**Desempenho e subamostragem.** Na GPU local (NVIDIA MX330, 2 GB), o OpenPose processa a
~1,2 s/frame (medido: `1.22s/frame` na execução da Figura 2). Processar os 5.191 frames de
PM_008 levaria ~1h45. Para viabilizar a execução local, o vídeo é **subamostrado para 1 a
cada 3 frames** (opção `--frame-step 3` do CLI; 10 fps efetivos, 1.731 frames), reduzindo o
tempo do OpenPose para os 45:20 medidos (o pipeline reporta o tempo no output e no
relatório) sem perda relevante para o movimento lento do agachamento. O mapeamento é
preservado: o índice `i` do keypoint corresponde ao frame original `3*i`, permitindo o
cruzamento correto com os rótulos (ver 3.8).

![Uso da GPU local durante a extração de pose](figures/screenshots/uso_gpu_local.png)

> **Figura 1.** Gerenciador de Tarefas durante a extração do PM_008. A MX330 opera em
> **90% de utilização a 71 °C**, com **1,4 dos 2,0 GB** de memória dedicada ocupados — é
> essa margem estreita de VRAM que justifica o `--net_resolution 320x176` da tabela acima:
> resoluções maiores não cabem. A carga está na GPU dedicada, não na integrada (Intel Iris
> Xe, 10%), confirmando que o OpenPose usa o dispositivo correto. O gargalo é a GPU, não a
> CPU (38%) — o que explica por que a subamostragem, e não o paralelismo, é a alavanca de
> desempenho disponível aqui.

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

O PM_006 (3.8) exercita essa estratégia no cenário mais difícil disponível — uma pessoa ao
fundo, luzes apagadas e câmera em meio-perfil: a seleção robusta preserva a separação entre
execuções corretas e incorretas, ainda que a margem caia de 10,5x para 2,8x em relação ao
vídeo capturado em condições favoráveis. Como os dois vídeos diferem em várias variáveis ao
mesmo tempo, essa queda não é atribuível isoladamente à presença da outra pessoa.

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

**Resultados quantitativos (PM_034).** O mesmo pipeline, sem qualquer ajuste de parâmetros,
foi aplicado a um exercício diferente (abdução de perna) e a um enquadramento frontal.
Sobre os 370 frames subamostrados (10 fps efetivos), o detector sinalizou 122 (33,0%) como
desvio, com cobertura de 100% em todas as juntas. Cruzando com os rótulos das 10 repetições:

| Classe da repetição | Taxa média de frames anômalos |
|:--|:--:|
| Correta (5 repetições) | 0,034 |
| Incorreta (5 repetições) | 0,358 |

A separação aqui é **muito mais nítida que no PM_008** — cerca de 10x entre as classes,
contra ~1,4x no agachamento. A explicação está na natureza do movimento: a abdução de perna
tem uma postura de referência estável (paciente em pé, de frente), então a mediana do vídeo
representa bem a execução correta e só a execução incorreta se afasta dela. Já o agachamento
é um movimento de grande amplitude, em que mesmo a execução correta percorre posturas
distantes da mediana — inflando a taxa das repetições corretas (ver a observação de método
acima). O resultado sugere que o método é mais discriminante em exercícios de amplitude
moderada em torno de uma postura estável, e que exercícios cíclicos amplos se beneficiariam
de uma referência por fase do movimento em vez de uma mediana global.

A articulação predominante foi o **ombro D** (48% dos instantes sinalizados), coerente com
o uso dos braços para compensar o equilíbrio durante a abdução.

**Resultados quantitativos (PM_006) — desempenho em condições adversas de captura.** O
PM_006 executa o mesmo exercício do PM_034, com a mesma estrutura de repetições e duração
equivalente, porém em condições de captura bem piores: **câmera em meio-perfil, luzes
apagadas e uma pessoa ao fundo** (além de outro sujeito e da perna oposta — ver a ressalva
em 3.3). Sobre os 348 frames subamostrados, o detector sinalizou 96 (27,6%):

| Classe da repetição | PM_034 (condições favoráveis) | PM_006 (condições adversas) |
|:--|:--:|:--:|
| Correta (5 repetições) | 0,034 | 0,175 |
| Incorreta (5 repetições) | 0,358 | 0,488 |
| **Razão incorreta/correta** | **10,5x** | **2,8x** |

O resultado relevante é que a separação **se mantém**: mesmo no pior cenário disponível, as
repetições incorretas concentram quase 3x mais desvios que as corretas, e o sinal continua
utilizável. A margem, porém, **cai mais de 3x** em relação ao vídeo favorável, e a queda vem
principalmente das repetições *corretas*, cuja taxa de anomalia sobe 5x (0,034 → 0,175) — a
degradação se manifesta como **falsos positivos** em execuções boas, não como incapacidade
de detectar as ruins. Para um sistema de triagem clínica, esse é o modo de falha menos
grave: perde-se especificidade, não sensibilidade.

A qualidade da estimação de pose acompanha a queda: a orelha esquerda é detectada em apenas
3,2% dos frames (contra cobertura de 100% em todas as juntas no PM_034), e a articulação
predominante muda para a **inclinação do tronco** (71% dos instantes sinalizados, contra
ombro D no PM_034).

> **O que este resultado não demonstra.** É tentador atribuir a degradação à pessoa ao
> fundo, mas os dois vídeos diferem simultaneamente em sujeito, perna, orientação de câmera,
> iluminação e presença de terceiros (3.3). A iluminação apagada e o ângulo em meio-perfil
> explicariam igualmente bem tanto a perda de cobertura das juntas quanto a mudança da
> articulação predominante para o tronco. O que se pode afirmar é que **o pipeline preserva
> o sinal discriminante sob condições adversas combinadas**; atribuir a queda a um fator
> específico exigiria um desenho experimental que variasse uma variável por vez.

**Consolidação dos três experimentos.**

| | PM_008 | PM_034 | PM_006 |
|:--|:--:|:--:|:--:|
| Exercício | agachamento (Ex6) | abdução de perna (Ex4) | abdução de perna (Ex4) |
| Condições de captura | luzes apagadas | favoráveis | adversas |
| Frames analisados | 1.731 | 370 | 348 |
| Frames com desvio | 576 (33,3%) | 122 (33,0%) | 96 (27,6%) |
| Taxa — corretas | 0,430 | 0,034 | 0,175 |
| Taxa — incorretas | 0,614 | 0,358 | 0,488 |
| Razão | 1,4x | 10,5x | 2,8x |
| Articulação predominante | joelho D (41%) | ombro D (48%) | tronco (71%) |
| Tempo — OpenPose | 45:20.089 | 10:06.450 | 09:38.986 |
| Tempo — análise | 00:15.541 | 00:21.878 | 00:02.829 |

Em todos os três, a **separação tem o sinal esperado** (incorretas > corretas), sem nenhum
ajuste de parâmetro entre execuções — o mesmo `contamination = 0.03`, o mesmo limiar
`|z| > 3.5` e o mesmo `random_state = 42`. Esse é o resultado central da entrega: o método
se sustenta em dois exercícios distintos, dois sujeitos e três condições de captura, com
parâmetros fixos.

A margem, porém, varia bastante (de 1,4x a 10,5x). Com três vídeos não é possível
quantificar a contribuição de cada fator, mas os dados são compatíveis com duas
influências: a **amplitude do movimento** (o agachamento, de grande amplitude, produz a
menor margem) e a **qualidade da captura** (o vídeo em condições adversas produz margem
intermediária). Ambas são hipóteses sugeridas pelos dados, não relações isoladas
experimentalmente. O tempo de análise escala com o número de frames; o tempo de extração
domina o custo em qualquer cenário.

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

### 3.10 Ferramenta de Linha de Comando (`cli.py`)

O `cli.py` executa o pipeline completo em um único comando — subamostragem, OpenPose,
ângulos, detecção de desvios, relatório, vídeo anotado e validação — e é a ferramenta usada
para processar lotes e para as execuções de referência deste relatório.

```bash
python -m src.video.cli --video data/video/rehab24-6/PM_008-Camera17-30fps.mp4 \
    --openpose-root tools/openpose --fps 30 --frame-step 3 --overlay \
    --segmentation data/video/rehab24-6/Segmentation.csv
```

**Parâmetros.** A origem dos dados é obrigatória e mutuamente exclusiva: ou `--video`
(extrai a pose antes) ou `--json-dir` (reaproveita keypoints já extraídos).

| Parâmetro | Padrão | Função |
|:--|:--|:--|
| `--video` | — | vídeo a processar; roda o OpenPose antes. Exclusivo com `--json-dir` |
| `--json-dir` | — | pasta com os `*_keypoints.json` já extraídos (pula o OpenPose) |
| `--openpose-root` | — | raiz do binário do OpenPose; **obrigatório** com `--video` |
| `--fps` | `30.0` | taxa do vídeo original, usada para converter frames em segundos |
| `--frame-step` | `1` | processa 1 a cada N frames. `3` reduz o tempo do OpenPose em ~3x e é o valor usado nas execuções locais; também mapeia os frames na validação |
| `--overlay` | desligado | gera o vídeo com esqueleto e desvios sobrepostos (requer `--video`) |
| `--segmentation` | — | CSV de rótulos do REHAB24-6; ativa a validação contra o ground-truth |
| `--video-id` | derivado do nome | id no `Segmentation.csv` (ex.: `PM_008-Camera17-30fps` → `PM_008`) |
| `--net-resolution` | `320x176` | resolução da rede do OpenPose; menor = menos VRAM |
| `--out` | `reports` | pasta de saída dos artefatos |

**Saída.** O terminal acompanha as fases numeradas (`[0/4]` a `[4/4]`) com barra de
progresso `tqdm` nas duas etapas longas (OpenPose e overlay), e fecha imprimindo os tempos
por fase no formato `mm:ss.mi`, o resultado da validação e o caminho de cada artefato
gerado.

![Execução do CLI sobre o vídeo PM_008](figures/screenshots/cli_PM_008-Camera17-30fps.png)

> **Figura 2.** Execução completa do CLI sobre o PM_008 (1.731 frames após subamostragem).
> A saída mostra as quatro fases, os tempos medidos — OpenPose 45:20.089, análise 00:15.541,
> overlay e validação somando +11:06.050, total fim-a-fim 56:42.061 — e a validação contra o
> ground-truth (corretas 0,430 · incorretas 0,614, separação OK). Ao final, lista os quatro
> artefatos gerados: relatório, gráfico, vídeo anotado e CSV de validação por repetição.

### 3.11 Interface Web de Demonstração (`app.py`)

Para uso pela **equipe clínica** e para a demonstração do desafio, o mesmo pipeline é
exposto em uma app web local (Gradio, `python -m src.video.app`, porta 7860). A app não
reimplementa nada: chama as mesmas funções do CLI e gera os mesmos artefatos em `reports/`.

A diferença está na linguagem. Como o público não é técnico, as etapas aparecem por extenso
("Aplicando o Modelo OpenPose", "Overlay (esqueleto sobre o vídeo)"), cada quadro de saída
tem uma legenda dizendo o que vai aparecer nele, e os tempos usam `mm:ss.mi`. Os controles
são quatro: o vídeo (lista montada a partir de `data/video/rehab24-6/*.mp4`), o `frame-step`,
a validação contra o ground-truth e o reaproveitamento de keypoints já extraídos.

A sequência abaixo mostra o fluxo completo do ponto de vista de quem usa a ferramenta.

![Vídeo original do PM_034, sem anotação](figures/screenshots/video_sem_overlay_PM_034-Camera17-30fps.png)

> **Figura 3 — antes.** O material de partida: o vídeo bruto do PM_034 (abdução de perna),
> como a equipe o receberia hoje. A avaliação depende inteiramente da observação visual do
> profissional, sem nenhuma medida objetiva de ângulo ou marcação de instantes suspeitos.

![Interface Gradio processando o PM_034](figures/screenshots/gradio_processando_PM_034-Camera17-30fps.png)

> **Figura 4 — durante.** A app em processamento, com o `frame-step` em 3 e o
> reaproveitamento de keypoints desligado (extração completa). A barra informa a etapa em
> linguagem corrente e o progresso real ("Aplicando o Modelo OpenPose — 104/370 frames"),
> e cada quadro já anuncia o que vai exibir ao terminar.

![Interface Gradio com o resultado do PM_034](figures/screenshots/gradio_finalizado_PM_034-Camera17-30fps.png)

> **Figura 5 — depois.** O mesmo instante da Figura 3, agora processado. À esquerda, os
> ângulos ao longo dos 37 s com as faixas rosa marcando os instantes de desvio; à direita, o
> vídeo com o esqueleto BODY_25 sobreposto, a borda vermelha sinalizando o frame como desvio
> e o osso do ângulo responsável destacado em vermelho (ombro, no frame 0). Abaixo — fora do
> enquadramento — seguem a validação contra o ground-truth e o relatório completo.

### 3.12 Testes

O pipeline é coberto por testes automatizados que usam **keypoints sintéticos** (sem
necessidade de OpenPose nem do dataset): validam o cálculo de ângulos, a detecção de
anomalias injetadas e a renderização do overlay.

---

## 4. Entrega 2 — Análise de Áudio (AWS)

### 4.1 Visão Geral

O pipeline de áudio processa gravações de consultas médicas e de fonação, transcreve a
fala, extrai os achados clínicos mencionados e mede biomarcadores vocais, produzindo um
relatório para a equipe médica.

```
consulta (.mp3) ──► S3 ──► Transcribe ──► diarização ──► Comprehend Medical ──┐
                                                                              ├─► relatório
vogal sustentada (.wav) ──► Praat/librosa ──► jitter, shimmer, HNR, MFCC ─────┘   bilíngue
```

### 4.2 Troca de Provedor: Azure → AWS

O enunciado sugere Azure Cognitive Services. O grupo não dispunha de cota no Azure for
Students e a instituição liberou o uso da **AWS**, para onde o pipeline foi migrado. A
equivalência é direta e, num ponto, favorável:

| Função | Azure (previsto) | AWS (implementado) |
|:--|:--|:--|
| Transcrição | Azure AI Speech | **Amazon Transcribe** |
| Entidades clínicas | Text Analytics for Health | **Amazon Comprehend Medical** |
| Tradução | — | **Amazon Translate** |

O Comprehend Medical devolve as entidades já **tipadas e qualificadas por traços**
(`NEGATION`, `PERTAINS_TO_FAMILY`, `HYPOTHETICAL`), o que carrega para o serviço parte da
lógica clínica que, de outro modo, teria de ser implementada localmente.

### 4.3 Duas Fontes de Dados, por Necessidade Técnica

A entrega usa **dois datasets**, e a divisão não é arbitrária:

| Dataset | Fornece | Alimenta |
|:--|:--|:--|
| Consultas simuladas (figshare, CC0) | fala clínica espontânea + transcrição humana | Transcribe → Comprehend Medical |
| Coswara (IISc, open-access) | fonação sustentada e respiração, com sintoma rotulado | biomarcadores (Praat/librosa) |

**Jitter e shimmer medem a perturbação ciclo a ciclo da vibração das pregas vocais** e só
são calculáveis sobre fonação sustentada — uma vogal mantida por alguns segundos. Em
conversa espontânea, com dois interlocutores e sobreposição de fala, não há ciclos
comparáveis. O Coswara tem essa fonação; as consultas, não.

Na direção oposta, a única fala do Coswara é a **contagem de números**, que não produz
linguagem clínica para o Comprehend Medical extrair. As consultas produzem. Nenhum dos
dois substitui o outro.

**Detalhes dos datasets.** As consultas somam 272 casos (213 respiratórios, 78,3%), MP3
16 kHz mono de 11-15 min, com transcrição humana revisada. Do Coswara foram usados 2 dos
45 lotes (1,7 GB de ~28 GB), com 9 gravações por participante e rótulos de qualidade por
escuta manual.

> **Definição do grupo de controle.** `covid_status == healthy` **não basta**: 121 dos
> 1.433 participantes assim declarados relatam algum sintoma, sendo 12 com dificuldade
> respiratória e 12 com fadiga — exatamente os sintomas que definem o grupo oposto. O
> controle exige, além do status, ausência de qualquer sintoma relatado.

### 4.4 Transcrição (`transcribe.py`)

O Transcribe **não aceita áudio na chamada**: o arquivo vai para o S3, inicia-se um job
assíncrono sobre a URI e busca-se o JSON do resultado. A diarização fica ligada
(`MaxSpeakerLabels=2`), o que permite separar médico e paciente.

**Resultado (RES0029, 6,7 min de consulta respiratória):**

| Métrica | Valor |
|:--|:--:|
| WER | **5,37%** |
| Substituições / inserções / deleções | 20 / 16 / 6 |
| Palavras na referência | 782 |
| Turnos identificados | 65 (referência humana: 69) |
| Tempo do job | 47 s |

O WER é medido contra a transcrição humana do próprio dataset, por distância de edição em
nível de palavra. **Hesitações ("um", "uh") são removidas dos dois lados**: o anotador
humano as transcreveu, o Transcribe as omite, e contá-las mediria a convenção de anotação,
não o reconhecimento. A diferença é grande e vale registrar — **5,37% sem hesitações
contra 13,30% com elas**.

**O que o WER não mostra.** A métrica trata todas as palavras como iguais, mas errar
`two`→`2` não tem o mesmo peso clínico que errar `chest pain`. Verificando termo a termo,
os **15 termos clínicos da consulta aparecem com contagem idêntica** nas duas transcrições
— `chest` 5/5, `pain` 10/10, `breath` 5/5, além de `cough`, `fever`, `headache`,
`dizziness`, `stabbing`, `nausea`, `vomiting`, `chills`. Nenhum ausente. Os erros
concentram-se em convenções de escrita (`i'd`→`i would`, `nope`→`no`) e em palavras
funcionais repetidas em trechos com hesitação.

### 4.5 Extração de Entidades Clínicas (`comprehend.py`)

O Comprehend Medical recebe **apenas a fala do paciente** e devolve entidades tipadas. A
restrição é essencial: as perguntas do médico ("any fever?", "do you have a cough?")
contêm termos clínicos, mas são hipóteses sendo investigadas, não achados do paciente.

Na origem humana, a separação vem dos rótulos `D:`/`P:` do dataset; na origem AWS, da
diarização do Transcribe. O papel do paciente é identificado por dois sinais independentes
— quem fala mais e quem *não* abre a consulta — e o código recusa-se a decidir se eles
discordarem, porque trocar os papéis inverteria todo o relatório.

**Resultado (RES0029):** 52 entidades, entre elas:

| Entidade | Traço | Leitura clínica |
|:--|:--|:--|
| `cough`, `infections` | NEGATION | o paciente **negou** |
| `diabetes` | PERTAINS_TO_FAMILY | ocorre em **familiar** |
| `alcohol`, `drugs`, `marijuana` | NEGATION | negativas de uso |
| `left side of my chest` | (anatomia) | localização do sintoma |

Ignorar os traços produziria um relatório listando como sintomas do paciente coisas que
ele negou ou que pertencem a um parente.

**Validação entre as duas origens.** Como existem a transcrição humana e a automática para
a mesma consulta, é possível responder o que o WER não responde: *os 5% de erro atrapalham
a extração clínica?* Extraindo das duas, **7 dos 9 achados foram recuperados**
(recall 0,778). A leitura literal, porém, subestima: os dois "perdidos" — `hurts` e
`impact` — são variantes lexicais de achados que **foram** recuperados (`pain`, `painful`,
`fell`, `fell off`). **Nenhum achado clinicamente distinto se perdeu.**

Em sentido oposto, o serviço extraiu `head was fine` como achado positivo, quando o
paciente afirmava justamente estar bem — falso positivo que fica registrado como
limitação.

### 4.6 Biomarcadores Acústicos (`biomarkers.py`)

Jitter, shimmer e HNR vêm do **Praat** (via `parselmouth`), implementação de referência
dessas medidas, calculadas ciclo a ciclo sobre os pulsos glotais. O rastreamento de F0 por
frame de bibliotecas como o librosa permitiria apenas uma aproximação, e usar o nome
"jitter" para ela seria impreciso; o librosa fica com os MFCC e a leitura de áudio.

**Resultado: negativo.** Com 30 participantes por grupo, sobre a vogal /a/ sustentada,
**nenhuma medida separou sintomáticos de saudáveis** (todos os p > 0,05, Mann-Whitney U).

A única diferença aparente — F0 mais baixa nos sintomáticos (158 vs 187 Hz) — é
**confundida por sexo**: a F0 mediana é 219,9 Hz em mulheres e 130,7 Hz em homens, e os
grupos estão desbalanceados (sintomáticos 21H/9M, saudáveis 17H/13M). Estratificando por
sexo, a diferença desaparece (p = 0,60 em homens, p = 0,13 em mulheres).

Replicando nas três vogais, o **shimmer é maior nos sintomáticos em todas elas**:

| Vogal | Sintomáticos | Saudáveis | p |
|:--|:--:|:--:|:--:|
| /a/ | 0,065 | 0,053 | 0,167 |
| /e/ | 0,063 | 0,051 | 0,751 |
| /o/ | 0,067 | 0,043 | **0,006** |

O resultado em /o/ mantém o sinal nos dois sexos (p = 0,028 em homens, p = 0,061 em
mulheres). Foram, porém, **15 testes** (5 medidas × 3 vogais), e o limiar de Bonferroni é
0,0033 — **o achado não sobrevive à correção para múltiplas comparações**. A consistência
de direção nas três vogais é sugestiva e justifica amostra maior; não constitui
biomarcador demonstrado.

**Interpretação do resultado negativo.** Duas explicações são plausíveis e não excludentes:
os sintomas do Coswara são **autorrelatados**, o que introduz ruído no rótulo; e jitter,
shimmer e HNR medem a **laringe**, sendo tradicionalmente aplicados a disfonias, enquanto
"dificuldade respiratória e fadiga" afetam sobretudo o sistema respiratório.

### 4.7 Relatório Clínico Bilíngue (`report.py`)

O relatório destina-se à equipe médica e é **bilíngue por necessidade**: o áudio-fonte é em
inglês, e traduzir sem mostrar o original impediria a conferência contra a gravação.
Cada termo e cada trecho aparecem no original seguido da tradução (Amazon Translate, com
cache por trecho).

Estrutura: achados relatados → sintomas negados → história familiar → trechos de apoio →
qualidade da transcrição. Cada achado vem acompanhado da **frase que o originou**, porque
`pain` isolado não informa se é torácica, intensa ou momentânea.

Três ajustes de **segurança clínica** foram necessários após ler a primeira saída como
leitor médico:

1. **Termos afirmados e negados na mesma consulta.** `pain` aparecia nas duas tabelas — o
   paciente tem dor torácica (queixa principal) e negou dor em outro local. Listar "dor:
   negada" ao lado da queixa principal poderia levar ao seu descarte. Havendo conflito, o
   achado afirmado prevalece e um aviso nomeia os termos ambíguos para verificação.
2. **História familiar entre os achados do paciente.** `diabetes` aparecia como achado
   próprio; o filtro excluía hipotéticos, mas não `PERTAINS_TO_FAMILY`.
3. **Tradução ambígua.** `drugs` era traduzido como "medicamentos", enquanto o tipo é
   `REC_DRUG_USE`; a tabela de negações passou a exibir o tipo.

O relatório informa também o **WER da transcrição que originou os achados**: extração
perfeita sobre transcrição ruim continua sendo informação ruim, e a equipe precisa desse
contexto para calibrar a confiança.

### 4.8 Controle de Custo e Credenciais

Transcribe, Comprehend Medical e Translate cobram por volume processado. Todo resultado é
**cacheado em disco** (`reports/transcriptions/`, `reports/entities/`,
`reports/translations.json`) e nenhum caso é reprocessado sem `--force`. O modo `--report`
recalcula métricas a partir do cache **sem tocar na AWS**, permitindo iterar sobre a
métrica sem custo. Os JSON brutos da AWS são versionados, de forma que os resultados podem
ser auditados sem credenciais.

As credenciais ficam em `~/.aws/credentials` (via `aws configure`); o `.env` guarda apenas
região e nome do bucket, sem segredos. O comando `python -m src.common.config` verifica o
ambiente antes de qualquer chamada paga.

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

## 6. Integração em Nuvem (AWS) e Fluxo de Alerta

O enunciado exige integração com serviços gerenciados em nuvem. O projeto usa **três
serviços da AWS**, todos na Entrega 2, com o processamento das demais modalidades
executado localmente.

### 6.1 Serviços Utilizados

| Serviço | Papel | Estado |
|:--|:--|:--|
| **Amazon S3** | armazenamento do áudio — o Transcribe não aceita upload direto | Em uso |
| **Amazon Transcribe** | transcrição da fala, com diarização de 2 falantes | Em uso |
| **Amazon Comprehend Medical** | extração de entidades clínicas tipadas (`DetectEntitiesV2`) | Em uso |
| **Amazon Translate** | tradução dos achados para o relatório bilíngue | Em uso |

Região: `us-east-1`. A escolha não é indiferente — o **Comprehend Medical não está
disponível em todas as regiões**, e o comando de verificação do ambiente confere isso
antes de qualquer chamada.

### 6.2 O que Roda Local, e por quê

| Modalidade | Processamento | Justificativa |
|:--|:--|:--|
| Vídeo (Entrega 1) | OpenPose local | Não há serviço gerenciado equivalente para pose clínica; a extração de keypoints é feita pelo binário do OpenPose |
| Anomalias (Entrega 3) | IsolationForest local | Ver nota abaixo |
| Biomarcadores (Entrega 2) | Praat/librosa local | Jitter e shimmer são medidas de sinal, não de linguagem; nenhum serviço gerenciado das nuvens avaliadas as oferece |

**Nota sobre serviços gerenciados de anomalia.** Os dois serviços que seriam o encaixe
natural para séries temporais foram descontinuados pelos respectivos provedores:

- **Azure Anomaly Detector** — não permite criar novos recursos desde 20 de setembro de
  2023, com retirada completa prevista.
- **Amazon Lookout for Metrics** — **suporte encerrado em 10 de outubro de 2025**; o
  serviço deixou de ser acessível pelo console e pela API, e a AWS orienta migrar para
  CloudWatch, OpenSearch, Redshift ML ou QuickSight.

A Entrega 3 permanece, portanto, com detecção local (IsolationForest). A decisão não
decorre de preferência técnica, mas do fato de que **os serviços gerenciados dedicados a
essa função foram retirados do mercado pelos dois provedores**. Uma alternativa gerenciada
ainda viável seria a detecção de anomalias do **Amazon CloudWatch**, voltada a métricas
operacionais — encaixe possível, porém menos direto para séries clínicas multivariadas com
rótulo de deterioração.

### 6.3 Evidência de Uso

Os serviços deixam rastro auditável no console da AWS, o que permite verificar a
integração de forma independente do código:

| Serviço | Onde verificar |
|:--|:--|
| Transcribe | Console → Amazon Transcribe → *Transcription jobs* (nome, status, idioma, horário) |
| S3 | Console → S3 → bucket → prefixo `consultations/` |
| Comprehend Medical | Console → CloudTrail → *Event history*, filtrando por `comprehendmedical.amazonaws.com` |
| Métricas do Transcribe | CloudWatch → namespace `AWS/Transcribe` (18 métricas, incluindo `AudioDurationTime`) |

O Comprehend Medical **não publica métricas no CloudWatch** (o namespace
`AWS/ComprehendMedical` permanece vazio); sua evidência de uso está no CloudTrail, que
registra cada chamada `DetectEntitiesV2`.

### 6.4 Fluxo de Alerta à Equipe Médica

**[Em desenvolvimento]** — a camada de fusão entre as três modalidades e o disparo
automático de alerta ainda não estão implementados. O desenho previsto consome o sinal de
anomalia de cada pipeline (desvio postural, achado clínico, anomalia em sinal vital) e
encaminha o alerta à equipe.

O relatório clínico bilíngue (4.7) já constitui a **saída legível** desse fluxo: reúne os
achados, sua origem na fala do paciente e a confiabilidade da transcrição que os produziu.
O que falta é a automação do disparo.

---

## 7. Decisões de Projeto e Justificativas

### 7.1 Vídeo: OpenPose como Extrator Externo (JSON), Análise em Python

Em vez de usar a API Python do OpenPose (que exige compilação em C++/CUDA), o binário é
invocado por linha de comando e grava keypoints em JSON. Todo o restante é Python puro. Isso
isola a dependência pesada, torna o pipeline portátil e simplifica os
testes (keypoints sintéticos).

### 7.2 Troca de Dataset: KIMORE → REHAB24-6

O KIMORE ficou indisponível (servidor fora do ar) e os *mirrors* traziam apenas esqueleto
(sem RGB), inviabilizando o OpenPose. O REHAB24-6 fornece RGB e rótulos de correção — melhor
para o objetivo — e é aberto no Zenodo.

### 7.3 Subamostragem de Frames para Viabilizar a GPU Local

A GPU local (MX330, 2 GB) processa a ~1,2 s/frame. Subamostrar 1 a cada 3 frames reduz o
tempo em ~3x, adequado ao movimento lento do agachamento, preservando o mapeamento de frames
para validação.

### 7.4 Detecção de Anomalias Local, por Indisponibilidade de Serviço Gerenciado

Nem o Azure Anomaly Detector nem o Amazon Lookout for Metrics podem ser provisionados
(ambos descontinuados — ver 6.2). Mantém-se o
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
│   ├── common/                 # config e utilitários compartilhados (AWS, caminhos)
│   ├── video/                  # Entrega 1 — análise de vídeo (OpenPose)
│   │   ├── keypoints.py        # parser dos JSON BODY_25
│   │   ├── posture.py          # ângulos articulares por frame
│   │   ├── anomaly.py          # detecção de desvios (z-score + IsolationForest)
│   │   ├── report.py           # relatório Markdown + gráfico
│   │   ├── overlay.py          # vídeo anotado (esqueleto + desvios)
│   │   ├── validate.py         # validação contra os rótulos do REHAB24-6
│   │   ├── run_openpose.py     # invocação do binário OpenPose (com progresso)
│   │   ├── cli.py              # pipeline fim-a-fim (barra de progresso no terminal)
│   │   └── app.py              # app web (Gradio) para o vídeo-demo
│   ├── audio/                  # Entrega 2 — análise de áudio (AWS)
│   └── anomaly/                # Entrega 3 — detecção de anomalias
│       ├── load_challenge2019.py   # loader + baseline (sinais vitais)
│       └── load_uci_har.py         # loader + baseline (movimentação)
├── data/                       # datasets baixados localmente (não versionado)
├── docs/                       # enunciado + guias (datasets, setup do OpenPose)
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
| Gradio | App web local de demonstração (Entrega 1) |
| AWS (Transcribe, Comprehend Medical, Translate, S3) | Serviços gerenciados da Entrega 2 |
| Praat (via `parselmouth`) | Jitter, shimmer e HNR (Entrega 2) |

### 9.2 Bibliotecas Python

| Biblioteca | Uso |
|:--|:--|
| `numpy`, `pandas` | Manipulação de dados e séries |
| `scikit-learn` | IsolationForest, imputação |
| `matplotlib`, `seaborn` | Gráficos e relatórios |
| `opencv-python` | Leitura de vídeo e overlay do esqueleto |
| `boto3` | Amazon Transcribe e Comprehend Medical (Entrega 2) |
| `librosa`, `soundfile` | Biomarcadores acústicos (Entrega 2) |
| `python-dotenv` | Carregamento de credenciais da nuvem |
| `pytest` | Testes automatizados |

---

## 10. Reprodutibilidade

### 10.1 Pré-requisitos

- Python 3.10+ e as dependências de `requirements.txt`
- OpenPose v1.7.0 (binário portátil) — ver `docs/openpose_setup.md`
- (Entrega 2) Conta AWS com acesso a Transcribe, Comprehend Medical e um bucket S3;
  credenciais em `.env` ou `~/.aws/credentials`

### 10.2 Instalação

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows (Git Bash)
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

**Determinismo e versões.** Os resultados deste relatório são reprodutíveis: o
`random_state = 42` é fixo em todos os detectores e as demais operações (mediana, MAD,
cálculo de ângulos) são determinísticas. Isso foi verificado na prática — os artefatos
foram apagados e os três vídeos reprocessados do zero, e as taxas de validação saíram
idênticas às da execução anterior.

O `requirements.txt` declara **pisos** de versão (`numpy>=1.24`), para instalar sem atrito
em qualquer máquina. O risco dessa escolha é que `random_state` fixa a *semente*, não o
*algoritmo*: se uma versão futura do `scikit-learn` alterar a implementação interna do
`IsolationForest`, os valores publicados aqui podem não se reproduzir.

Por isso o repositório traz também **`requirements-lock.txt`**, com as versões exatas sob
as quais os números deste relatório foram medidos. Para auditar os resultados:

```bash
pip install -r requirements-lock.txt
```

**Teste de estabilidade entre versões.** O pipeline foi recalculado a partir dos mesmos
keypoints em dois ambientes independentes, com versões distintas das bibliotecas
numéricas:

| Biblioteca | Ambiente A | Ambiente B |
|:--|:--:|:--:|
| scikit-learn | 1.7.2 | 1.9.0 |
| pandas | 2.3.3 | 3.0.3 |
| scipy | 1.16.3 | 1.18.0 |
| numpy | 2.4.6 | 2.4.6 |

Os dois produziram resultados **idênticos bit a bit** nos três vídeos — 96, 576 e 122
frames sinalizados — com os CSVs de validação por repetição inalterados. O resultado é
relevante porque as versões divergem justamente no `scikit-learn`, de onde vem o
`IsolationForest`: na prática, a detecção se mostrou estável a uma mudança de versão menor
dessa biblioteca, e não apenas à fixação da semente.

### 10.3 Execução — Entrega 1 (Análise de Vídeo)

```bash
# Vídeo -> OpenPose -> relatório + vídeo anotado + validação, num comando:
#   --frame-step 3  subamostra (GPU fraca)
#   --overlay       gera o vídeo anotado
#   --segmentation  valida contra os rótulos correto/incorreto
python -m src.video.cli --video data/video/rehab24-6/PM_034-Camera17-30fps.mp4 \
    --openpose-root tools/openpose --fps 30 --frame-step 3 --overlay \
    --segmentation data/video/rehab24-6/Segmentation.csv

# A partir de JSONs já extraídos
python -m src.video.cli --json-dir reports/json/PM_034 --fps 30

# App web local de demonstração (abre em http://localhost:7860)
python -m src.video.app
```

A tabela completa de parâmetros do CLI está na **seção 3.10**; a interface web e seu fluxo
de uso, na **seção 3.11**.

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
| GPU local fraca (MX330, 2 GB) | OpenPose lento (~1,2 s/frame) | Subamostragem (`--frame-step`) |
| Datasets de modalidades distintas | Não há paciente comum entre vídeo, áudio e vitais | Prática acadêmica padrão; documentado |
| Serviços gerenciados de anomalia descontinuados (Azure Anomaly Detector e Amazon Lookout for Metrics) | Sem opção gerenciada direta para séries clínicas | Detecção local (IsolationForest); avaliar CloudWatch |
| Biomarcadores sem separação estatística | Sintomas autorrelatados e medidas laríngeas para sintoma respiratório (4.6) | Amostra maior; medidas sobre a respiração, não só a fonação |
| Fusão multimodal e alerta automático ausentes | As três modalidades não convergem num alerta único | Implementar a camada de fusão (6.4) |
| Detecção não-supervisionada | Limiares definidos empiricamente | Calibração com os rótulos disponíveis |
| Referência = mediana global do vídeo | Em movimentos de grande amplitude (agachamento), a execução correta também se afasta da mediana e a separação cai para 1,4x (3.8) | Referência por fase do movimento em vez de mediana única |
| Sensibilidade às condições de captura | Em condições adversas (pouca luz, meio-perfil, pessoa ao fundo), a margem cai de 10,5x para 2,8x, por falsos positivos em execuções corretas (3.8) | Rastreamento de identidade entre frames; máscara da região de interesse; normalização por iluminação |

### 11.2 Trabalhos Futuros

1. Implementar a camada de fusão multimodal e o disparo automático de alerta (6.4)
2. Adotar uma referência **por fase do movimento** (em vez da mediana global) para recuperar
   a separação em exercícios de grande amplitude — a limitação mais clara medida em 3.8
3. Implementar a camada de fusão e o fluxo de alerta em nuvem
4. Calibrar os limiares de anomalia com os *ground-truths* disponíveis
5. Processar mais vídeos/exercícios do REHAB24-6 para robustez (hoje: 3 experimentos,
   2 exercícios, 2 sujeitos)
6. **Isolar os fatores de degradação** com pares de vídeos que variem uma variável por vez
   (iluminação, orientação da câmera, presença de terceiros) — o REHAB24-6 tem metadados
   para montar esse desenho, que a comparação atual (3.3) não permite
7. Avaliar modelos temporais (LSTM/autoencoder) para as séries de sinais vitais

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

6. Amazon Web Services. Amazon Transcribe, Amazon Comprehend Medical e Amazon Translate — documentação. [docs.aws.amazon.com](https://docs.aws.amazon.com/)
7. Boersma, P., & Weenink, D. *Praat: doing phonetics by computer.* [praat.org](https://www.fon.hum.uva.nl/praat/)
