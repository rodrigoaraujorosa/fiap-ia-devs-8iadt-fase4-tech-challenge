# 🎙️ Entrega 2 — Análise de Áudio (AWS)

Processar áudios de consultas médicas, transcrever a fala e extrair os achados clínicos
relatados pelo paciente, produzindo um relatório para a equipe médica.

## Pipeline

```
                                        ┌─► Comprehend Medical ──► achados ─┐
consulta (.mp3) ──► S3 ──► Transcribe ──┤                                   ├─► relatório
                          (diarização)  └─► Comprehend ────────► sentimento ┘   bilíngue
```

1. **Amazon Transcribe** — transcrição da fala, com diarização de 2 falantes.
2. **Amazon Comprehend Medical** — entidades clínicas tipadas (sintomas, anatomia,
   medicações) com traços como `NEGATION` e `PERTAINS_TO_FAMILY`.
3. **Amazon Comprehend** — sentimento do relato, geral e por turno de fala.
4. **Amazon Translate** — tradução dos achados para o relatório bilíngue.

> **Provedor de nuvem: AWS.** O plano original usava Azure Cognitive Services, mas não
> havia cota disponível. Com a liberação da AWS, o pipeline passou a usar Transcribe +
> Comprehend Medical — este último é um encaixe melhor que o equivalente da Azure para o
> objetivo da entrega, por extrair entidades clínicas já tipadas.

🔑 Credenciais via `.env` (modelo em `.env.example`) ou `~/.aws/credentials`, carregadas por
`src/common/config.py`. O Transcribe **exige que o áudio esteja no S3** — não aceita upload
direto na chamada.

## Módulos

| Módulo | Papel |
|---|---|
| `cli.py` | **único ponto de entrada**: roda as quatro etapas, ou os modos sem custo |
| `consultations.py` | loader do dataset: lista casos, separa turnos por falante, isola a fala do paciente |
| `transcribe.py` | upload ao S3, job do Transcribe, diarização e medição de WER contra a referência humana |
| `comprehend.py` | entidades clínicas, sentimento e comparação entre as duas transcrições |
| `report.py` | relatório bilíngue para a equipe médica |
| `cache.py` | cache em disco dos resultados pagos, comum aos módulos |

> `transcribe.py`, `comprehend.py` e `report.py` são **bibliotecas, sem CLI própria** —
> mesma organização da Entrega 1, onde apenas `cli.py` e `app.py` são executáveis.

## Dataset: consultas médicas simuladas

| Item | Valor |
|---|---|
| Fonte | [figshare, DOI 10.6084/m9.figshare.16550013.v1](https://doi.org/10.6084/m9.figshare.16550013.v1) |
| Conteúdo | 272 consultas em formato OSCE, com áudio e **transcrição humana revisada** |
| Áudio | MP3 16 kHz mono, 11-15 min por consulta |
| Especialidades | 213 respiratórias (78,3%), 46 musculoesqueléticas, 13 outras |
| Licença | CC0 (domínio público) |

A **transcrição humana** é ground-truth: permite medir o erro do Transcribe em vez de
apenas exibir o resultado.

> ⚠️ **Armadilha de codificação.** Dois dos 213 casos respiratórios (`RES0002` e
> `RES0054`) estão em UTF-16, o resto em UTF-8. Ler tudo como UTF-8 não levanta erro —
> devolve texto corrompido, e o caso aparece silenciosamente com zero turnos de fala. O
> loader detecta a codificação pelo BOM.

## Uso

### Tudo de uma vez

```bash
# pipeline completo: transcrição -> entidades -> sentimento -> relatório
python -m src.audio.cli --case RES0091

# vários casos, com estatística do WER ao final
python -m src.audio.cli --cases RES0091 RES0142 RES0094 --out reports/wer_consultations.csv

# o que seria cobrado, sem executar nada
python -m src.audio.cli --case RES0091 --dry-run
```

### Sem chamar a nuvem

```bash
# estatísticas do dataset, para escolher os casos
python -m src.audio.consultations --root data/audio/consultas --summary
python -m src.audio.consultations --root data/audio/consultas --case RES0001 --patient-only

# recalcula o WER do que já está em cache
python -m src.audio.cli --report --out reports/wer_consultations.csv

# inspeciona as entidades já extraídas de um caso
python -m src.audio.cli --show-entities RES0091
```

## Por que só a fala do paciente

As perguntas do médico ("any fever?", "do you have a cough?") contêm termos clínicos, mas
são hipóteses sendo investigadas — não achados do paciente. Incluí-las produziria uma
lista de sintomas que o paciente nunca relatou.

Na transcrição humana a separação vem dos rótulos `D:`/`P:`; na da AWS, da diarização. O
papel do paciente é identificado por dois sinais independentes — quem fala mais e quem
*não* abre a consulta — e o código recusa-se a decidir se eles discordarem.

## Onde ficam os artefatos

```
data/audio/consultas/Data/         # dataset (gitignored)
  Audio Recordings/RES0001.mp3     #   áudio da consulta
  Clean Transcripts/RES0001.txt    #   transcrição humana (ground-truth)

reports/                           # resultados (versionados)
  transcriptions/RES0001.json      #   resposta bruta do Transcribe
  entities/RES0001__human.json     #   entidades da transcrição humana
  entities/RES0001__aws.json       #   entidades da transcrição automática
  entities/RES0001__sentiment.json #   sentimento (geral e por turno)
  translations.json                #   cache do Translate, compartilhado entre casos
  audio_RES0001.md                 #   relatório clínico bilíngue
  wer_consultations.csv            #   métricas de transcrição de todos os casos
```

Os JSON brutos da AWS são versionados de propósito: permitem recalcular as métricas e
auditar os números do relatório **sem credenciais e sem custo**.

## Controle de custo

Transcribe, Comprehend Medical, Comprehend e Translate cobram por volume. Todo resultado é **cacheado**
(`reports/transcriptions/`, `reports/entities/`, `reports/translations.json`) e nenhum caso
é reprocessado sem `--force`. O modo `--report` recalcula métricas do cache **sem chamar a
AWS**. Os JSON brutos são versionados, permitindo auditar os resultados sem credenciais.
