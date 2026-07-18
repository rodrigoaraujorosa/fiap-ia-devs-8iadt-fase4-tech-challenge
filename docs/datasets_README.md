# Kit de datasets — Tech Challenge Fase 4

Datasets **públicos de download imediato** (sem credenciamento), por modalidade.

| Entrega | Subtarefa | Dataset | Acesso | Tamanho |
|---|---|---|---|---|
| 1 — Análise de Vídeo | Análise postural (reabilitação) | REHAB24-6 | Aberto (Zenodo) | ~2,7 GB (vídeos) |
| 2 — Análise de Áudio | Fala clínica (transcrição + entidades) | Consultas médicas simuladas | Aberto (figshare, CC0) | 986 MB |
| 2 — Análise de Áudio | Biomarcadores acústicos | Coswara | Open-access (não-comercial) | ~28 GB total, **usamos 1,7 GB** |
| 3 — Detecção de Anomalias | Séries temporais de sinais vitais | PhysioNet/CinC Challenge 2019 (Sepsis) | Aberto | ~42 MB |
| 3 — Detecção de Anomalias | Padrões de movimentação do paciente | UCI HAR (Human Activity Recognition) | Aberto | ~60 MB |
| 3 — Detecção de Anomalias | Evolução de prescrições | Synthea (sintético) | Aberto | variável |

**Por que a Entrega 2 usa dois datasets.** Não é redundância: cada um cobre uma metade que
o outro não tem. Jitter, shimmer e F0 são medidas de perturbação ciclo a ciclo da vibração
das pregas vocais e exigem **fonação sustentada** — não se calculam de forma confiável em
conversa espontânea, com dois interlocutores e sobreposição de fala. O Coswara tem vogais
sustentadas e respiração, com sintoma rotulado por participante, mas sua única fala é
**contar números**, o que não gera linguagem clínica para o Comprehend Medical extrair. As
consultas simuladas dão exatamente essa linguagem, e não têm fonação sustentada.

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

Use `src/audio/consultas.py`:

```bash
python -m src.audio.consultas --root data/audio/consultas --resumo
python -m src.audio.consultas --root data/audio/consultas --caso RES0001 --paciente
```

---

## 2. Coswara — biomarcadores acústicos (Entrega 2)

Respiração (profunda/rasa), tosse (pesada/rasa), vogais sustentadas (/a/, /e/, /o/) e
contagem de números (rápida/normal): **nove gravações por participante**, com metadados de
sintomas e comorbidades.

**Download seletivo (o repositório inteiro tem ~28 GB):**

```bash
mkdir -p data/audio/coswara
BASE=https://raw.githubusercontent.com/iiscleap/Coswara-Data/master

# 1) Metadados (~2 MB) — permitem desenhar o recorte antes de baixar áudio
curl -L -o data/audio/coswara/combined_data.csv       "$BASE/combined_data.csv"
curl -L -o data/audio/coswara/csv_labels_legend.json  "$BASE/csv_labels_legend.json"

# 2) Rótulos de qualidade por gravação (escuta manual: 0 ruim, 1 boa, 2 excelente)
mkdir -p data/audio/coswara/annotations
for s in breathing-deep breathing-shallow cough-heavy cough-shallow \
         counting-fast counting-normal vowel-a vowel-e vowel-o; do
  curl -L -o "data/audio/coswara/annotations/${s}_labels.csv" \
    "$BASE/annotations/${s}_labels.csv"
done

# 3) CSV de cada lote — mapeia participante -> lote (necessário: record_date NÃO serve)
mkdir -p data/audio/coswara/folder_csv
# (um curl por lote; ver src/audio/dataset.py)

# 4) Áudio: só os lotes escolhidos, em partes de 100 MB
mkdir -p data/audio/coswara/raw/20220224
for p in aa ab ac ad ae af ag ah ai aj ak al am an ao; do
  curl -L -o "data/audio/coswara/raw/20220224/20220224.tar.gz.$p" \
    "$BASE/20220224/20220224.tar.gz.$p"
done
python -m src.audio.dataset --root data/audio/coswara --extrair 20220224
```

Página oficial: https://github.com/iiscleap/Coswara-Data (open-access, **não-comercial**).
Artigo: https://arxiv.org/abs/2005.10548

**Lotes usados neste trabalho:**

| Lote | Tamanho | Papel |
|---|---|---|
| `20220224` | 1.369 MB (15 partes) | rico em sintomáticos |
| `20210406` | 317 MB (4 partes) | reforça o grupo de controle |

**Metadados relevantes** (legenda completa em `csv_labels_legend.json`): `bd` = dificuldade
respiratória, `ftg` = fadiga — exatamente os sintomas do enunciado do desafio. Também
`cough`, `fever`, `st` (dor de garganta), `asthma`, `cld` (doença pulmonar crônica),
`covid_status`.

> ⚠️ **`covid_status == healthy` não basta para o grupo de controle.** 121 dos 1.433
> participantes assim declarados relatam algum sintoma — 12 deles justamente dificuldade
> respiratória ou fadiga, que definem o grupo oposto. O loader exige, além do status,
> **ausência de qualquer sintoma relatado**.

> 💡 Os `.tar.gz` vêm fatiados em partes de 100 MB porque o GitHub limita o tamanho de
> arquivo. Elas precisam ser concatenadas antes de descompactar — `--extrair` faz isso.

Use `src/audio/dataset.py`:

```bash
python -m src.audio.dataset --root data/audio/coswara --resumo
python -m src.audio.dataset --root data/audio/coswara --coorte \
    --lotes 20220224 20210406 --por-grupo 30
```

---

## 3. PhysioNet/CinC Challenge 2019 — sinais vitais

**Download (escolha uma opção):**

```bash
# Opção A: wget recursivo (Linux/Mac)
wget -r -N -c -np https://physionet.org/files/challenge-2019/1.0.0/

# Opção B: zips diretos
wget https://physionet.org/static/published-projects/challenge-2019/training_setA.zip
wget https://physionet.org/static/published-projects/challenge-2019/training_setB.zip
unzip training_setA.zip -d challenge2019/
```

Página oficial: https://physionet.org/content/challenge-2019/1.0.0/

**Formato:** 1 arquivo `.psv` por paciente (pipe-delimited, cabeçalho presente).
Cada linha = 1 hora de internação. 40 variáveis + `SepsisLabel`.

**Colunas (ordem oficial):**
- Sinais vitais (8): `HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2`
- Laboratório (26): `BaseExcess, HCO3, FiO2, pH, PaCO2, SaO2, AST, BUN, Alkalinephos, Calcium, Chloride, Creatinine, Bilirubin_direct, Glucose, Lactate, Magnesium, Phosphate, Potassium, Bilirubin_total, TroponinI, Hct, Hgb, PTT, WBC, Fibrinogen, Platelets`
- Demografia (6): `Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS`
- Rótulo: `SepsisLabel` (0/1) — deterioração clínica, serve de ground-truth.

Use `src/anomaly/load_challenge2019.py` para carregar e rodar o baseline.

---

## 4. UCI HAR — movimentação do paciente

**Download:**

```bash
wget https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip
unzip human+activity+recognition+using+smartphones.zip
unzip "UCI HAR Dataset.zip" -d uci_har/
```

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

Use `src/anomaly/load_uci_har.py` para carregar e rodar o baseline.

**Enquadramento como anomalia de movimentação:** trate atividades esperadas
(ex.: LAYING/SITTING durante internação) como "normal" e sinalize transições
bruscas ou atividades inesperadas (ex.: queda ~ pico de aceleração) como anomalia.

---

## 5. Evolução de prescrições

Sem fonte pública aberta granular (a boa vem do MIMIC, que exige credenciamento).
Duas saídas defensáveis na banca:

**Opção A — Synthea (sintético, recomendado):**
```bash
git clone https://github.com/synthetichealth/synthea.git
cd synthea && ./run_synthea -p 1000
# gera output/csv/medications.csv com START, STOP, PATIENT, CODE, DESCRIPTION
```
Site: https://synthetichealth.github.io/synthea/ — gera registros clínicos realistas,
incluindo série temporal de prescrições por paciente.

**Opção B — variável derivada:** trate intervenções/doses registradas ao longo
das horas no Challenge 2019 como a série monitorada e detecte mudanças bruscas.

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
python -m src.audio.consultas --root data/audio/consultas --resumo
python -m src.audio.dataset   --root data/audio/coswara  --resumo

# Entrega 3 — anomalias
python src/anomaly/load_challenge2019.py --data ./data/anomaly/challenge2019
python src/anomaly/load_uci_har.py --data "./data/anomaly/uci_har/UCI HAR Dataset"
```

> Os datasets ficam em `data/`, que é gitignored — cada um precisa baixar os seus.
