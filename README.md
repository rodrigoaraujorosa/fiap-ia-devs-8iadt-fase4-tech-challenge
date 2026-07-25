# 🏥 Sistema de Monitoramento Hospitalar Multimodal

> **Tech Challenge — Fase 4** · PosTech FIAP · IA para DEVs (turma 8IADT)

![Status](https://img.shields.io/badge/status-3%20entregas%20conclu%C3%ADdas-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10%20ou%20superior-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![OpenPose](https://img.shields.io/badge/OpenPose-BODY__25-00FFFF)
![AWS](https://img.shields.io/badge/AWS-Transcribe%20%C2%B7%20Comprehend%20Medical-FF9900?logo=amazonaws&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

Plataforma que monitora continuamente pacientes em ambiente hospitalar / UTI a partir de
**dados multimodais** — vídeo, áudio e séries temporais de sinais vitais — para identificar
**sinais precoces de risco** e emitir **alertas automáticos** para a equipe médica.

---

## 🎯 Visão geral das entregas

O projeto é composto por **3 entregas técnicas independentes**, unidas pelo cenário clínico.
Cada modalidade usa um **dataset público de download imediato** (prática acadêmica padrão,
não são os mesmos pacientes nas 3 fontes).

| # | Entrega | Objetivo | Dataset | Modelos / Serviços |
|---|---------|----------|---------|--------------------|
| 1 | 🎥 **Análise de Vídeo** | Detectar movimentos/eventos fora do padrão em vídeos clínicos | REHAB24-6 (reabilitação, RGB + rótulos correto/incorreto) | OpenPose (BODY_25) |
| 2 | 🎙️ **Análise de Áudio** | Transcrever consultas, extrair achados clínicos e analisar o sentimento do relato | Consultas médicas simuladas (figshare) | Amazon Transcribe · Comprehend Medical · Comprehend · Translate |
| 3 | 📈 **Detecção de Anomalias** | Anomalias em sinais vitais, prescrições e movimentação, com alerta à equipe | PhysioNet Challenge 2019 · UCI HAR | IsolationForest · regra de degrau (dose) |

---

## 📂 Estrutura do projeto

```
fiap-ia-devs-8iadt-fase4-tech-challenge/
├── src/
│   ├── common/config.py             # caminhos, credenciais AWS e verificação do ambiente
│   ├── video/                       # Entrega 1 — análise de vídeo (OpenPose, local)
│   │   ├── run_openpose.py          #   invoca o binário do OpenPose
│   │   ├── keypoints.py             #   parser BODY_25 + seleção da pessoa principal
│   │   ├── posture.py anomaly.py    #   ângulos articulares; desvios (z-score ∪ IsolationForest)
│   │   ├── validate.py report.py    #   validação contra o ground-truth; relatório + gráfico
│   │   ├── overlay.py               #   vídeo anotado com esqueleto e desvios
│   │   ├── cli.py                   #   pipeline fim-a-fim
│   │   └── app.py                   #   app web de demonstração (Gradio)
│   ├── audio/                       # Entrega 2 — análise de áudio (AWS)
│   │   ├── consultations.py         #   loader do dataset; separa médico/paciente
│   │   ├── transcribe.py            #   S3 + Amazon Transcribe + medição de WER
│   │   ├── comprehend.py            #   Comprehend Medical (entidades) e Comprehend (sentimento)
│   │   ├── report.py                #   relatório clínico bilíngue (Amazon Translate)
│   │   ├── cache.py                 #   cache dos resultados pagos
│   │   ├── cli.py                   #   pipeline fim-a-fim
│   │   └── app.py                   #   app web de demonstração (Gradio)
│   ├── dashboard/                   # painel unificado — as 3 entregas em abas
│   │   └── app.py                   #   camada de apresentação (Gradio), porta 7863
│   └── anomaly/                     # Entrega 3 — detecção de anomalias (local)
│       ├── movement.py              #   movimentação do paciente (UCI HAR)
│       ├── vitals.py                #   sinais vitais de UTI (Challenge 2019)
│       ├── prescriptions.py         #   evolução de doses (FiO2, variável derivada)
│       ├── alerts.py                #   fila de plantão, priorizada por confiabilidade
│       ├── report.py                #   relatório para a equipe médica + figuras
│       ├── cli.py                   #   pipeline fim-a-fim
│       └── app.py                   #   painel de plantão (Gradio)
├── models/                          # detectores treinados (.joblib, NÃO versionado)
├── data/                            # datasets baixados localmente (NÃO versionado)
│   ├── video/rehab24-6/             #   REHAB24-6 (vídeos + Segmentation.csv)
│   ├── audio/consultas/             #   consultas médicas simuladas
│   └── anomaly/                     #   Challenge 2019 e UCI HAR
├── reports/                         # resultados e relatório técnico
│   ├── TECHNICAL_REPORT_FASE4.md    #   o relatório da fase
│   ├── anomalias.md                 #   saída da Entrega 3 para a equipe médica
│   ├── relatorio_PM_*.md            #   relatórios de desvio postural (Entrega 1)
│   ├── validacao_PM_*.csv           #   validação por repetição (Entrega 1)
│   ├── audio_RES*.md audio_MSK*.md  #   relatórios clínicos bilíngues (Entrega 2)
│   ├── wer_consultations.csv        #   métricas de transcrição (Entrega 2)
│   ├── transcriptions/ entities/    #   respostas brutas da AWS (cache, versionado)
│   ├── translations.json            #   cache do Amazon Translate
│   └── figures/                     #   gráficos gerados + screenshots/ do relatório
├── docs/
│   ├── datasets_README.md           # download e schema de cada dataset
│   └── openpose_setup.md            # instalação do binário do OpenPose
├── tests/                           # 55 testes, sem exigir datasets nem credenciais
│   ├── test_video.py                #   9 testes, keypoints sintéticos
│   ├── test_anomaly.py              #   28 testes, séries sintéticas
│   ├── test_audio_app.py            #   8 testes, trava de custo da app da Entrega 2
│   └── test_dashboard.py            #   10 testes, montagem do painel unificado
├── requirements.txt                 # pisos de versão
├── requirements-lock.txt            # versões exatas, para auditar os resultados
├── .env.example                     # modelo de configuração da AWS (sem segredos)
├── pyproject.toml
├── LICENSE                          # MIT (código); datasets têm licença própria
└── README.md
```

> Os JSON brutos devolvidos pela AWS ficam **versionados** em `reports/transcriptions/` e
> `reports/entities/`. Isso permite recalcular as métricas da Entrega 2 — e auditar os
> números do relatório — **sem credenciais da AWS e sem custo**.

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
#   Para reproduzir exatamente os números do relatório, use o lock:
#   pip install -r requirements-lock.txt

# 3. (Entrega 2) Configurar a AWS — ver a seção "Configuração da AWS" abaixo
cp .env.example .env
```

> ⚠️ **Ative o ambiente virtual antes de qualquer comando deste README.** Todos os
> `python -m src...` assumem o `.venv` ativo. Rodar com o Python do sistema pode até
> funcionar, mas usa outras versões das bibliotecas — foi o que aconteceu durante o
> desenvolvimento, e o `scikit-learn` chegou a divergir em uma versão menor entre os dois
> ambientes. Para conferir qual interpretador está ativo:
>
> ```bash
> python -c "import sys; print(sys.prefix)"   # deve apontar para .../.venv
> ```

📥 As instruções de **download de cada dataset** estão em
[`docs/datasets_README.md`](docs/datasets_README.md).

---

## ☁️ Configuração da AWS (necessária para a Entrega 2)

A Entrega 2 usa **serviços gerenciados em nuvem** e, por isso, **não roda sem uma conta
AWS própria**. As Entregas 1 e 3 são inteiramente locais e não precisam de nada disto.

### 1️⃣ Conta e permissões

Crie uma conta em [aws.amazon.com](https://aws.amazon.com/) e um usuário IAM com acesso
programático. As permissões mínimas são:

| Serviço | Para quê | Ações necessárias |
|---|---|---|
| **Amazon S3** | o Transcribe **não aceita upload direto** — o áudio precisa estar no S3 | `s3:PutObject`, `s3:GetObject`, `s3:ListBucket` |
| **Amazon Transcribe** | transcrever a fala das consultas | `transcribe:StartTranscriptionJob`, `transcribe:GetTranscriptionJob` |
| **Amazon Comprehend Medical** | extrair entidades clínicas da transcrição | `comprehendmedical:DetectEntitiesV2` |
| **Amazon Comprehend** | analisar o sentimento do relato | `comprehend:DetectSentiment`, `comprehend:BatchDetectSentiment` |

As políticas gerenciadas `AmazonS3FullAccess`, `AmazonTranscribeFullAccess` e
`ComprehendMedicalFullAccess` cobrem tudo, são mais amplas que o necessário, aceitável
para um trabalho acadêmico, **não** para produção.

### 2️⃣ Região

Use **`us-east-1`**. O Comprehend Medical **não existe em todas as regiões** — em
particular, confirme antes de escolher `sa-east-1` (São Paulo). O comando de verificação
do passo 5 checa isso.

### 3️⃣ Credenciais

Instale a [AWS CLI](https://aws.amazon.com/cli/) e rode:

```bash
aws configure
# AWS Access Key ID     : (do seu usuário IAM)
# AWS Secret Access Key : (idem)
# Default region name   : us-east-1
# Default output format : json
```

Isso grava as chaves em `~/.aws/credentials`, de onde o `boto3` as lê automaticamente.

> 🔒 **Não coloque chaves no `.env`.** Ele guarda apenas região e nome do bucket. Manter o
> segredo em um só lugar (`~/.aws/credentials`, fora do repositório) reduz a chance de
> vazamento. O `.env` é gitignored, mas o hábito de duplicar segredo é o que vaza.

### 4️⃣ Bucket S3

```bash
aws s3 mb s3://SEU-BUCKET --region us-east-1
aws s3 ls                                    # confirme o NOME COMPLETO
```

> ⚠️ **Atenção ao nome.** Se você criar o bucket pelo console, a AWS pode acrescentar um
> sufixo com o id da conta e a região — `meu-bucket` vira
> `meu-bucket-123456789012-us-east-1-xx`. Use sempre o nome como aparece em `aws s3 ls`,
> senão o Transcribe falha com `NoSuchBucket` só na hora de rodar.

Preencha o `.env` (copiado de [`.env.example`](.env.example)):

```bash
AWS_REGION=us-east-1
AWS_S3_BUCKET=seu-bucket-completo-aqui
```

### 5️⃣ Verificação

```bash
python -m src.common.config
```

O comando confere, em ordem, as variáveis do `.env`, a credencial, a autenticação (via
`sts:GetCallerIdentity`), o acesso ao bucket e a existência dos endpoints do Transcribe e
do Comprehend Medical na região. Sai com código `0` se estiver tudo pronto e aponta o
passo que falhou caso contrário:

```
Verificação do ambiente AWS
  região             : us-east-1
  bucket             : seu-bucket-completo-aqui
  credencial         : encontrada
  autenticação (STS) : OK (conta ...1234)
  bucket S3          : acessível
  Transcribe         : disponível em us-east-1
  Comprehend Medical : disponível em us-east-1

ambiente pronto.
```

Nenhuma credencial é impressa; do identificador da conta aparecem só os 4 últimos dígitos.

---

## 🧪 Como executar

### 🏥 Painel unificado — as três entregas em uma tela

```bash
python -m src.dashboard.app        # abre em http://localhost:7863
```

Uma aba por modalidade, num endereço só. É a forma mais rápida de ver o sistema inteiro
funcionando — e as apps individuais continuam disponíveis nas portas de sempre (7860,
7862 e 7861).

> As abas **não fundem os dados**. As quatro fontes descrevem populações distintas, sem
> nenhum indivíduo em comum, e o rodapé do painel declara isso. Detalhes em
> [`src/dashboard/README.md`](src/dashboard/README.md).

### Entrega 1 — Análise de Vídeo (OpenPose)

Requer o binário do OpenPose ([`docs/openpose_setup.md`](docs/openpose_setup.md)) e um
vídeo do REHAB24-6. Um comando roda tudo: OpenPose → análise → relatório → vídeo anotado →
validação contra o ground-truth.

```bash
python -m src.video.cli --video data/video/rehab24-6/PM_034-Camera17-30fps.mp4 \
    --openpose-root tools/openpose --fps 30 --frame-step 3 --overlay \
    --segmentation data/video/rehab24-6/Segmentation.csv
```

- `--frame-step 3` subamostra o vídeo (acelera o OpenPose em GPU fraca)
- `--overlay` gera o vídeo com o esqueleto e os desvios marcados
- `--segmentation` valida os desvios contra os rótulos correto/incorreto

Há também uma **app web local** para demonstração (`python -m src.video.app`), que mostra o
progresso, o gráfico, o relatório e o vídeo com o esqueleto sobreposto. Detalhes em
[`src/video/README.md`](src/video/README.md).

### Entrega 2 — Análise de Áudio (AWS)

Requer conta AWS configurada (ver a seção acima) e o dataset de consultas em
`data/audio/consultas/`. Um comando roda o pipeline inteiro: upload ao S3 → transcrição →
entidades clínicas → sentimento → relatório bilíngue.

```bash
python -m src.audio.cli --case RES0029
```

- `--dry-run` mostra o que seria cobrado, **sem executar nada**
- `--cases A B C` processa em lote e reporta média e desvio do WER
- `--report` recalcula as métricas do cache, **sem chamar a AWS**
- `--no-translate` gera o relatório sem usar o Amazon Translate

O relatório sai em `reports/audio_<caso>.md`, **bilíngue**: cada achado e cada trecho
citado aparecem no original em inglês seguidos da tradução para o português — o áudio-fonte
é em inglês, e a equipe precisa poder conferir contra a gravação.

Há também uma **app web local** (`python -m src.audio.app`), que reproduz a consulta,
exibe os achados clínicos e o relatório bilíngue na tela.

> 💰 As etapas de nuvem são pagas por volume. Todo resultado é **cacheado**, e nenhum caso
> é reprocessado sem `--force`. Na app, a chamada paga é **bloqueada por padrão** e o
> seletor marca quais casos já estão em cache — com eles, a app roda **sem credenciais da
> AWS**. Detalhes em [`src/audio/README.md`](src/audio/README.md).

### Entrega 3 — Detecção de Anomalias

```bash
# 1. Treina os dois detectores no padrão de normalidade e salva em models/
python -m src.anomaly.cli --train --limit 5000

# 2. Monitoramento de UM indivíduo que o modelo não viu (é a demonstração)
python -m src.anomaly.cli --monitor p000188        # vitais + prescrições
python -m src.anomaly.cli --monitor-subject 2      # movimentação

# 3. Avaliação completa + relatório em reports/anomalias.md
python -m src.anomaly.cli --limit 5000
python -m src.anomaly.cli --only movement
```

Roda **inteiramente local** — não chama a nuvem e não custa nada. Detecção
não-supervisionada (IsolationForest); os rótulos do dataset entram só na avaliação.

---

## 📦 Datasets

| Modalidade | Dataset | Acesso | Link |
|------------|---------|--------|------|
| Vídeo | REHAB24-6 (reabilitação física, RGB) | Aberto (Zenodo, ~2,7 GB) | https://zenodo.org/records/13305826 |
| Áudio | Consultas médicas simuladas | Aberto (CC0) | https://doi.org/10.6084/m9.figshare.16550013.v1 |
| Sinais vitais | PhysioNet/CinC Challenge 2019 | Aberto (~42 MB) | https://physionet.org/content/challenge-2019/1.0.0/ |
| Movimentação | UCI HAR | Aberto (~60 MB) | https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones |
| Prescrições | variável derivada da `FiO2` (Challenge 2019) | — sem dataset extra | ver [`docs/datasets_README.md`](docs/datasets_README.md) §4 |

---

## 📚 Créditos e Citação

O **dataset REHAB24-6** é usado na Entrega 1 sob licença **CC BY-NC 4.0** (uso acadêmico,
não-comercial). Ao usar o dataset, cite:

> Černek, A., Sedmidubsky, J., Budikova, P., Jánošová, M., Katzer, L., & Procházka, M. (2024).
> *REHAB24-6: A multi-modal dataset of physical rehabilitation exercises* (v1) [Data set].
> Zenodo. https://doi.org/10.5281/zenodo.13305826

Artigo associado:

> Černek, A., Sedmidubsky, J., Budikova, P. (2024). *REHAB24-6: Physical Therapy Dataset for
> Analyzing Pose Estimation Methods.* 17th International Conference on Similarity Search and
> Applications (SISAP), Springer.

Os demais datasets e suas licenças estão em [`docs/datasets_README.md`](docs/datasets_README.md).

---

## 👥 Equipe

PosTech FIAP — IA para DEVs · Turma 8IADT · Grupo 30 do Tech Challenge Fase 4.
