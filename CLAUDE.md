# Tech Challenge Fase 4 — PosTech FIAP IA para DEVs

Contexto de handoff para o Claude Code. Projeto de **sistema de monitoramento hospitalar
multimodal** com 3 entregas técnicas independentes, unidas pelo cenário "paciente em
ambiente hospitalar / UTI". Cada modalidade usa um dataset público distinto (prática
padrão acadêmica; não existem os mesmos pacientes nas 3 fontes).

**Restrição de prazo:** poucos dias para entregar. Por isso todos os datasets escolhidos
são de **download imediato, sem credenciamento** (MIMIC-IV foi descartado — exige curso
CITI + Data Use Agreement, leva semanas).

---

## Entrega 1 — Análise de Vídeo

Detectar movimentos/eventos fora do padrão em vídeos clínicos.

**Dataset: KIMORE** (KInematic Assessment of MOvement and Clinical Scores)
- 78 sujeitos: 44 saudáveis + 34 com disfunções motoras
- Vídeo RGB, depth e skeleton (joint positions); 5 exercícios de reabilitação p/ dor lombar
- Traz clinical scores → ground-truth para validar desvios posturais
- Link: https://vrai.dii.univpm.it/content/kimore-dataset
- Alternativa (movimento correto/incorreto rotulado): **UI-PRMD**
  https://webpages.uidaho.edu/ui-prmd/

**Modelos:** OpenPose (análise postural sobre o skeleton/RGB).
**Se precisar cumprir YOLOv8 explicitamente** (detecção de objetos/áreas críticas):
KIMORE não tem objetos. Usar **Cholec80** + extensão **Cholec80-Boxes** (15.691 frames
com bounding boxes de 7 instrumentos cirúrgicos). Atenção: muda o cenário de fisioterapia
para cirurgia — decidir qual caminho seguir no vídeo.

## Entrega 2 — Análise de Áudio

Detectar alterações vocais/respiratórias em áudios de consulta.

**Dataset: Coswara**
- 2.635 indivíduos, 9 categorias de som (respiração rápida/lenta, tosse profunda/rasa,
  vogais sustentadas, dígitos falados) + metadados de sintomas e comorbidades
- Melhor encaixe para "dificuldades respiratórias e cansaço" (fala + respiração no mesmo sujeito)
- Open-access (não-comercial). Link: https://github.com/iiscleap/Coswara-Data
- Alternativas focadas em voz patológica: **Saarbrücken Voice Database**
  (https://stimmdb.coli.uni-saarland.de/) ou **VOICED/PhysioNet** (208 gravações)

**Stack pedido:** Azure Speech-to-Text (transcrição da fala) + Azure Text Analytics
(termos críticos e sentimento sobre a transcrição). Biomarcadores acústicos sobre o áudio bruto.

## Entrega 3 — Detecção de Anomalias

Séries temporais de sinais vitais, evolução de prescrições e padrões de movimentação.

**Sinais vitais → PhysioNet/CinC Challenge 2019 (Sepsis)** — ABERTO, ~42 MB
- 40.336 pacientes de UTI, séries horárias, 8 sinais vitais + 26 labs + demografia + SepsisLabel
- 1 arquivo `.psv` por paciente (pipe-delimited, cabeçalho presente), 1 linha = 1 hora
- SepsisLabel (0/1) = ground-truth de deterioração
- Link: https://physionet.org/content/challenge-2019/1.0.0/
- Colunas: HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2 (vitais) | BaseExcess, HCO3, FiO2,
  pH, PaCO2, SaO2, AST, BUN, Alkalinephos, Calcium, Chloride, Creatinine, Bilirubin_direct,
  Glucose, Lactate, Magnesium, Phosphate, Potassium, Bilirubin_total, TroponinI, Hct, Hgb,
  PTT, WBC, Fibrinogen, Platelets (labs) | Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS
  (demografia) | SepsisLabel
- Alternativa: PhysioNet Challenge 2012 (12.000 estadias, 37 variáveis)

**Movimentação do paciente → UCI HAR** — ABERTO, ~60 MB
- Vetores de 561 features (acelerômetro + giroscópio) + 6 atividades:
  WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING
- Sinais brutos em Inertial Signals/ (janelas de 128 timesteps, 50 Hz)
- Link: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- Enquadramento: repouso (LAYING/SITTING/STANDING) = normal; movimento inesperado
  (ex.: queda) = anomalia
- OBS: o link direto de download da UCI muda às vezes; se falhar, usar o mirror no Kaggle
  ("UCI HAR Dataset") — estrutura de pastas idêntica

**Evolução de prescrições** — sem fonte pública aberta granular (a boa vem do MIMIC, credenciado)
- Opção A (recomendada): **Synthea** — gerador sintético open-source, produz
  output/csv/medications.csv com START, STOP, PATIENT, CODE, DESCRIPTION.
  https://github.com/synthetichealth/synthea
- Opção B: variável derivada — tratar intervenções/doses do Challenge 2019 como série
  monitorada e detectar mudanças bruscas
- Documentar na banca que esta subtarefa usa dados sintéticos por indisponibilidade
  de fonte pública aberta

---

## Arquivos já prontos (copiar para a raiz do projeto)

- `datasets_README.md` — instruções de download (wget/unzip) + schemas completos
- `load_challenge2019.py` — loader dos .psv + baseline IsolationForest sobre os vitais;
  gera gráfico por paciente com `--patient p000001`. TESTADO (dados sintéticos).
- `load_uci_har.py` — loader do UCI HAR (com dedup de nomes de features duplicados) +
  baseline IsolationForest treinado só em atividades de repouso. TESTADO.

Instalar deps: `pip install pandas numpy scikit-learn matplotlib`

## Decisões pendentes

1. Vídeo: seguir OpenPose/KIMORE (postura, fisioterapia) OU YOLOv8/Cholec80 (instrumentos,
   cirurgia)? Os dois exigem datasets diferentes.
2. Prescrições: Synthea (sintético) vs. variável derivada do Challenge 2019.
