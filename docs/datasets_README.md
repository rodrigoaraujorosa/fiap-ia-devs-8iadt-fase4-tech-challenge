# Kit de datasets — Tech Challenge Fase 4

Datasets **públicos de download imediato** (sem credenciamento), por modalidade.

| Entrega | Subtarefa | Dataset | Acesso | Tamanho |
|---|---|---|---|---|
| 1 — Análise de Vídeo | Análise postural (reabilitação) | REHAB24-6 | Aberto (Zenodo) | ~2,7 GB (vídeos) |
| 3 — Detecção de Anomalias | Séries temporais de sinais vitais | PhysioNet/CinC Challenge 2019 (Sepsis) | Aberto | ~42 MB |
| 3 — Detecção de Anomalias | Padrões de movimentação do paciente | UCI HAR (Human Activity Recognition) | Aberto | ~60 MB |
| 3 — Detecção de Anomalias | Evolução de prescrições | Synthea (sintético) | Aberto | variável |

> Entrega 2 — Análise de Áudio: **Coswara** (open-access) — a documentar quando a entrega iniciar.

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

## 1. PhysioNet/CinC Challenge 2019 — sinais vitais

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

Use `load_challenge2019.py` para carregar e rodar o baseline.

---

## 2. UCI HAR — movimentação do paciente

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

Use `load_uci_har.py` para carregar e rodar o baseline.

**Enquadramento como anomalia de movimentação:** trate atividades esperadas
(ex.: LAYING/SITTING durante internação) como "normal" e sinalize transições
bruscas ou atividades inesperadas (ex.: queda ~ pico de aceleração) como anomalia.

---

## 3. Evolução de prescrições

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

```bash
pip install pandas numpy scikit-learn matplotlib
python load_challenge2019.py --data ./challenge2019
python load_uci_har.py --data "./uci_har/UCI HAR Dataset"
```
