# Kit de datasets — Tech Challenge Fase 4

Datasets **públicos de download imediato** (sem credenciamento), por modalidade.

| Entrega | Subtarefa | Dataset | Acesso | Tamanho |
|---|---|---|---|---|
| 1 — Análise de Vídeo | Análise postural (reabilitação) | REHAB24-6 | Aberto (Zenodo) | ~2,7 GB (vídeos) |
| 2 — Análise de Áudio | Consultas médicas (transcrição + entidades clínicas) | Consultas médicas simuladas | Aberto (figshare, CC0) | 986 MB |
| 3 — Detecção de Anomalias | Séries temporais de sinais vitais | PhysioNet/CinC Challenge 2019 (Sepsis) | Aberto | ~42 MB |
| 3 — Detecção de Anomalias | Padrões de movimentação do paciente | UCI HAR (Human Activity Recognition) | Aberto | ~60 MB |
| 3 — Detecção de Anomalias | Evolução de prescrições | Synthea (sintético) | Aberto | variável |

A Entrega 2 usa consultas médicas reais em formato simulado, com **fala clínica
espontânea** — o paciente descrevendo sintomas em linguagem natural, que é a matéria-prima
do Amazon Comprehend Medical. A transcrição humana que acompanha cada consulta serve de
ground-truth para medir o erro do Amazon Transcribe.

---

## 0. REHAB24-6 — vídeo de reabilitação (Entrega 1)

**Download (Zenodo, sem login):**

```bash
# vídeos RGB (2 câmeras, 30 fps) + rótulos de execução correta/incorreta
curl -L -o data/video/rehab24-6/videos.zip \
  "https://zenodo.org/records/13305826/files/videos.zip?download=1"
curl -L -o data/video/rehab24-6/Segmentation.csv \
  "https://zenodo.org/records/13305826/files/Segmentation.csv?download=1"
# extrair só o vídeo desejado do zip (ex.: Ex6/PM_008 ou Ex4/PM_034)
```

Página oficial: https://zenodo.org/records/13305826 (licença CC BY-NC 4.0, uso acadêmico).

**Conteúdo:** 6 exercícios (abdução de braço, arm VW, flexões, abdução de perna, lunge,
agachamento), 65 vídeos, 1.072 repetições rotuladas como **correta/incorreta** com o
intervalo de frames em `Segmentation.csv` (ground-truth para validar os desvios). O
OpenPose extrai os keypoints do RGB; ver `docs/openpose_setup.md` e `src/video/README.md`.

---

## 1. Consultas médicas simuladas — fala clínica (Entrega 2)

*A dataset of simulated patient-physician medical interviews with a focus on respiratory
cases* — consultas em formato OSCE, com áudio e **transcrição humana revisada**.

**Download (figshare, sem login):**

```bash
mkdir -p data/audio/consultas
curl -L -o data/audio/consultas/Data.zip \
  "https://ndownloader.figshare.com/files/30598530"
cd data/audio/consultas && unzip Data.zip     # ~986 MB
```

Página oficial: https://doi.org/10.6084/m9.figshare.16550013.v1 (licença **CC0**).

**Conteúdo:** 272 consultas, 11-15 min cada, MP3 **16 kHz mono** (formato que o Amazon
Transcribe aceita direto). Distribuição medida no dataset:

| Especialidade | Casos |
|---|---|
| Respiratório | 213 (78,3%) |
| Musculoesquelético | 46 (16,9%) |
| Gastrointestinal | 6 (2,2%) |
| Cardíaco | 5 (1,8%) |
| Dermatológico / Geral | 1 cada |

> O artigo do dataset informa 214 casos respiratórios (78,7%); a contagem sobre os arquivos
> baixados dá **213**. Usamos o número medido.

**Estrutura:**
```
Data/Audio Recordings/RES0001.mp3     # áudio da consulta
Data/Clean Transcripts/RES0001.txt    # transcrição humana, turnos marcados D: / P:
```

O prefixo do nome indica a especialidade (`RES`, `MSK`, `GAS`, `CAR`, `DER`, `GEN`).

**Papel na entrega:** a transcrição humana é **ground-truth** — permite medir o erro do
Amazon Transcribe em vez de apenas exibir o resultado. As falas do paciente (`P:`) vão
para o Comprehend Medical; as do médico não, porque descrevem perguntas e não achados
clínicos do paciente.

> ⚠️ **Armadilha de codificação.** 2 dos 213 casos respiratórios (`RES0002` e `RES0054`)
> estão em **UTF-16**; o resto em UTF-8. Ler tudo como UTF-8 não levanta erro — devolve
> texto corrompido, e o caso aparece silenciosamente com zero turnos de fala. O loader
> detecta a codificação pelo BOM.

Use `src/audio/consultations.py`:

```bash
python -m src.audio.consultations --root data/audio/consultas --summary
python -m src.audio.consultations --root data/audio/consultas --case RES0001 --patient-only
```

---

## 2. PhysioNet/CinC Challenge 2019 — sinais vitais

**Download — use o mirror S3, é o caminho rápido:**

```bash
mkdir -p data/anomaly/challenge2019
aws s3 sync --no-sign-request \
  s3://physionet-open/challenge-2019/1.0.0/training/ data/anomaly/challenge2019/
```

O bucket é público (`--no-sign-request` dispensa credenciais) e traz os dois conjuntos
de treino: `training_setA/` (20.336 pacientes) e `training_setB/` (20.000).

> ⚠️ **Os zips do PhysioNet não existem mais.** `training_setA.zip` e `training_setB.zip`
> em `physionet.org/static/published-projects/...` respondem **404** — os arquivos hoje
> são servidos individualmente, um `.psv` por paciente. Baixar por HTTP significa 40.336
> requisições; o `wget -r -N -c -np https://physionet.org/files/challenge-2019/1.0.0/`
> ainda funciona, mas é muito mais lento que o `aws s3 sync`.

Página oficial: https://physionet.org/content/challenge-2019/1.0.0/

**Formato:** 1 arquivo `.psv` por paciente (pipe-delimited, cabeçalho presente).
Cada linha = 1 hora de internação. 40 variáveis + `SepsisLabel`.

**Colunas (ordem oficial):**
- Sinais vitais (8): `HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2`
- Laboratório (26): `BaseExcess, HCO3, FiO2, pH, PaCO2, SaO2, AST, BUN, Alkalinephos, Calcium, Chloride, Creatinine, Bilirubin_direct, Glucose, Lactate, Magnesium, Phosphate, Potassium, Bilirubin_total, TroponinI, Hct, Hgb, PTT, WBC, Fibrinogen, Platelets`
- Demografia (6): `Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS`
- Rótulo: `SepsisLabel` (0/1) — deterioração clínica, serve de ground-truth.

> ⚠️ **`EtCO2` tem 0% de cobertura.** A coluna consta do schema oficial mas nunca é
> medida no training set A. O loader a descarta automaticamente — sem isso, entra no
> modelo como constante zero e só dilui a distância entre as amostras.

Use `python -m src.anomaly.cli` (ver `src/anomaly/README.md`).

---

## 3. UCI HAR — movimentação do paciente

**Download:**

```bash
mkdir -p data/anomaly/uci_har && cd data/anomaly/uci_har
curl -L -o har.zip \
  "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
unzip -q har.zip && unzip -q "UCI HAR Dataset.zip"   # o zip vem aninhado
rm -rf __MACOSX har.zip "UCI HAR Dataset.zip"
```

O download traz um zip **dentro** de outro: o externo contém `UCI HAR Dataset.zip` e um
`.names`. Extrair só o externo deixa a pasta sem os dados.

Página oficial: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

**Formato:** vetores de 561 features (acelerômetro + giroscópio) já extraídas.
6 atividades: `WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING`.
Sinais brutos em `Inertial Signals/` (janelas de 128 timesteps, 50 Hz).

Estrutura:
```
UCI HAR Dataset/
  activity_labels.txt      # id -> nome da atividade
  features.txt             # nomes das 561 colunas
  train/X_train.txt        # 7352 x 561
  train/y_train.txt        # rótulo de atividade
  train/subject_train.txt  # id do sujeito
  test/  (mesma estrutura)
```

Use `python -m src.anomaly.cli --only movement`.

**Enquadramento como anomalia de movimentação:** o modelo é treinado **só** com as
atividades de repouso (`LAYING`, `SITTING`, `STANDING`), que é o esperado em leito, e
nunca vê marcha no treino. As três atividades de marcha funcionam como ground-truth de
anomalia no teste.

---

## 4. Evolução de prescrições — variável derivada (decidido)

Não existe fonte pública aberta e granular de prescrições hospitalares: a base de
referência é o MIMIC-IV, que exige curso CITI e Data Use Agreement. **Não há dataset
novo a baixar nesta subtarefa.**

**Decisão do grupo: variável derivada do próprio Challenge 2019, usando a `FiO2`**
(fração inspirada de oxigênio). Ao contrário dos demais campos do dataset, que são
*medições* do paciente, a FiO2 é um **valor prescrito e titulado pela equipe** — sua
série ao longo das horas é uma série de doses, que é o objeto da subtarefa. É também o
campo de melhor cobertura entre os não-vitais (14,2%).

Anomalia = degrau de 0,15 ou mais entre coletas consecutivas. Só **aumentos** alertam:
reduzir a FiO2 é desmame, sinal de melhora.

A alternativa considerada e descartada foi o **Synthea** (gerador sintético que produz
`medications.csv` com `START, STOP, PATIENT, CODE, DESCRIPTION`). Seria mais fiel ao
enunciado, mas exige Java JDK e geração local, e as duas opções precisam da mesma
ressalva na banca — nenhuma é prescrição real de paciente real. A variável derivada
mantém uma única fonte de dados na entrega e preserva o mesmo ground-truth
(`SepsisLabel`).

**Ressalva a declarar:** é uma *proxy* de prescrição, não a prescrição registrada em
prontuário. O escalonamento de oxigênio é uma decisão terapêutica real, mas cobre apenas
um eixo do que uma base de prescrições traria.

---

## Instalação

As dependências de todos os loaders estão em `requirements.txt` (raiz do projeto):

```bash
pip install -r requirements.txt
```

Com o `.venv` ativo, a partir da raiz do projeto:

```bash
# Entrega 1 — vídeo
python -m src.video.cli --video data/video/rehab24-6/PM_034-Camera17-30fps.mp4     --openpose-root tools/openpose --fps 30 --frame-step 3 --overlay     --segmentation data/video/rehab24-6/Segmentation.csv

# Entrega 2 — áudio
python -m src.audio.consultations --root data/audio/consultas --summary
python -m src.audio.transcribe --report

# Entrega 3 — anomalias (roda local, não custa nada)
python -m src.anomaly.cli                       # três subtarefas + relatório
python -m src.anomaly.cli --only movement       # uma subtarefa
python -m src.anomaly.cli --limit 5000          # mais pacientes do Challenge 2019
```

> Os datasets ficam em `data/`, que é gitignored — cada um precisa baixar os seus.
