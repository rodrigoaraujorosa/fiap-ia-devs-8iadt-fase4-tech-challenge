# Kit de datasets — Detecção de Anomalias (Tech Challenge Fase 4)

Datasets **públicos de download imediato** (sem credenciamento). Cobrem as 3 subtarefas
da entrega de Detecção de Anomalias.

| Subtarefa | Dataset | Acesso | Tamanho |
|---|---|---|---|
| Séries temporais de sinais vitais | PhysioNet/CinC Challenge 2019 (Sepsis) | Aberto | ~42 MB |
| Padrões de movimentação do paciente | UCI HAR (Human Activity Recognition) | Aberto | ~60 MB |
| Evolução de prescrições | Synthea (sintético) ou variável derivada do Challenge 2019 | Aberto | variável |

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
