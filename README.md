# 🏥 Sistema de Monitoramento Hospitalar Multimodal

> **Tech Challenge — Fase 4** · PosTech FIAP · IA para DEVs (turma 8IADT)

![Status](https://img.shields.io/badge/status-em%20desenvolvimento-yellow)
![Python](https://img.shields.io/badge/Python-3.10%20ou%20superior-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![OpenPose](https://img.shields.io/badge/OpenPose-BODY__25-00FFFF)
![Azure](https://img.shields.io/badge/Azure-Cognitive%20Services-0078D4?logo=microsoftazure&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

Plataforma que monitora continuamente pacientes em ambiente hospitalar / UTI a partir de
**dados multimodais** — vídeo, áudio e séries temporais de sinais vitais — para identificar
**sinais precoces de risco** e emitir **alertas automáticos** para a equipe médica.

---

## 🎯 Visão geral das entregas

O projeto é composto por **3 entregas técnicas independentes**, unidas pelo cenário clínico.
Cada modalidade usa um **dataset público de download imediato** (prática acadêmica padrão —
não são os mesmos pacientes nas 3 fontes).

| # | Entrega | Objetivo | Dataset | Modelos / Serviços |
|---|---------|----------|---------|--------------------|
| 1 | 🎥 **Análise de Vídeo** | Detectar movimentos/eventos fora do padrão em vídeos clínicos | REHAB24-6 (reabilitação, RGB + rótulos correto/incorreto) | OpenPose (BODY_25) |
| 2 | 🎙️ **Análise de Áudio** | Detectar alterações vocais/respiratórias (fadiga, disartria) | Coswara | Azure Speech-to-Text · Azure Text Analytics · biomarcadores acústicos |
| 3 | 📈 **Detecção de Anomalias** | Anomalias em sinais vitais, prescrições e movimentação | PhysioNet Challenge 2019 · UCI HAR · Synthea | IsolationForest (baseline) |

---

## 📂 Estrutura do projeto

```
fiap-ia-devs-8iadt-fase4-tech-challenge/
├── src/                      # código-fonte
│   ├── common/               # config e utilitários compartilhados
│   ├── video/                # Entrega 1 — análise de vídeo (OpenPose)
│   │   ├── keypoints.py posture.py anomaly.py   # parser, ângulos, desvios
│   │   ├── report.py overlay.py validate.py     # relatório, vídeo anotado, validação
│   │   └── run_openpose.py cli.py               # OpenPose + pipeline fim-a-fim
│   ├── audio/                # Entrega 2 — análise de áudio
│   └── anomaly/              # Entrega 3 — detecção de anomalias
│       ├── load_challenge2019.py   # loader + baseline (sinais vitais)
│       └── load_uci_har.py         # loader + baseline (movimentação)
├── data/                     # datasets baixados localmente (não versionado)
│   ├── video/  audio/  anomaly/
├── docs/                     # enunciado do desafio + guia de datasets
│   ├── 8IADT - Fase 4 - Tech challenge.pdf
│   └── datasets_README.md
├── notebooks/                # exploração e prototipagem
├── reports/                  # relatório técnico e figuras geradas
│   └── figures/
├── tests/                    # testes automatizados
├── requirements.txt
├── pyproject.toml
├── .env.example              # modelo de credenciais Azure
├── LICENSE                   # MIT (código); datasets têm licença própria
└── README.md
```

---

## 🚀 Setup

```bash
# 1. Criar e ativar ambiente virtual
python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# Linux / Mac
source .venv/bin/activate

# 2. Instalar dependências
pip install -r requirements.txt

# 3. (Entrega 2) Configurar credenciais Azure
cp .env.example .env
#   e preencha AZURE_SPEECH_KEY / AZURE_LANGUAGE_KEY ...
```

📥 As instruções de **download de cada dataset** estão em
[`docs/datasets_README.md`](docs/datasets_README.md).

---

## 🧪 Como executar

### Entrega 1 — Análise de Vídeo (OpenPose)

Requer o binário do OpenPose ([`docs/openpose_setup.md`](docs/openpose_setup.md)) e um
vídeo do REHAB24-6. Um comando roda tudo: OpenPose → análise → relatório → vídeo anotado →
validação contra o ground-truth.

```bash
python -m src.video.cli --video data/video/rehab24-6/PM_006-Camera17-30fps.mp4 \
    --openpose-root tools/openpose --fps 30 --frame-step 3 --overlay \
    --segmentation data/video/rehab24-6/Segmentation.csv
```

- `--frame-step 3` subamostra o vídeo (acelera o OpenPose em GPU fraca)
- `--overlay` gera o vídeo com o esqueleto e os desvios marcados
- `--segmentation` valida os desvios contra os rótulos correto/incorreto

Sem GPU adequada, use o notebook [`notebooks/openpose_rehab24-6_colab.ipynb`](notebooks/openpose_rehab24-6_colab.ipynb).
Detalhes em [`src/video/README.md`](src/video/README.md).

### Entrega 3 — Detecção de Anomalias

```bash
# Sinais vitais (PhysioNet Challenge 2019)
python src/anomaly/load_challenge2019.py --data ./data/anomaly/challenge2019
python src/anomaly/load_challenge2019.py --data ./data/anomaly/challenge2019 --patient p000001

# Movimentação do paciente (UCI HAR)
python src/anomaly/load_uci_har.py --data "./data/anomaly/uci_har/UCI HAR Dataset"
```

---

## 📦 Datasets

| Modalidade | Dataset | Acesso | Link |
|------------|---------|--------|------|
| Vídeo | REHAB24-6 (reabilitação física, RGB) | Aberto (Zenodo, ~2,7 GB) | https://zenodo.org/records/13305826 |
| Áudio | Coswara | Open-access | https://github.com/iiscleap/Coswara-Data |
| Sinais vitais | PhysioNet/CinC Challenge 2019 | Aberto (~42 MB) | https://physionet.org/content/challenge-2019/1.0.0/ |
| Movimentação | UCI HAR | Aberto (~60 MB) | https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones |
| Prescrições | Synthea (sintético) | Aberto | https://github.com/synthetichealth/synthea |

---

## 📝 Entregáveis da Fase 4

- ✅ **Repositório Git** com código-fonte completo
- 📄 **Relatório técnico** ([`reports/`](reports/)): fluxo multimodal, modelos por tipo de
  dado, resultados e exemplos de anomalias detectadas
- 🎬 **Vídeo** (até 15 min, YouTube/Vimeo) demonstrando o processamento multimodal, a
  detecção/resposta a anomalias, a integração Azure e o fluxo de alerta à equipe médica

---

## 👥 Equipe

PosTech FIAP — IA para DEVs · Turma 8IADT · Grupo do Tech Challenge Fase 4.
