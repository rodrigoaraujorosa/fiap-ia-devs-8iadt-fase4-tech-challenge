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
| `consultations.py` | loader do dataset: lista casos, separa turnos por falante, isola a fala do paciente |
| `transcribe.py` | upload ao S3, job do Transcribe, diarização e medição de WER contra a referência humana |
| `comprehend.py` | entidades clínicas, sentimento e comparação entre as duas transcrições |
| `report.py` | relatório bilíngue para a equipe médica |

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

```bash
# estatísticas do dataset (não chama a AWS)
python -m src.audio.consultations --root data/audio/consultas --summary

# só as falas do paciente de um caso
python -m src.audio.consultations --root data/audio/consultas --case RES0001 --patient-only

# transcrição (CUSTA — cacheia em reports/transcriptions/)
python -m src.audio.transcribe --cases RES0029

# métricas do que já está em cache, sem tocar na AWS
python -m src.audio.transcribe --report --out reports/wer_consultations.csv

# entidades clínicas; --compare mede humano vs AWS
python -m src.audio.comprehend --cases RES0029
python -m src.audio.comprehend --cases RES0029 --compare

# sentimento do relato (geral e por turno)
python -m src.audio.comprehend --cases RES0029 --sentiment

# relatório final para a equipe médica
python -m src.audio.report --case RES0029
```

## Por que só a fala do paciente

As perguntas do médico ("any fever?", "do you have a cough?") contêm termos clínicos, mas
são hipóteses sendo investigadas — não achados do paciente. Incluí-las produziria uma
lista de sintomas que o paciente nunca relatou.

Na transcrição humana a separação vem dos rótulos `D:`/`P:`; na da AWS, da diarização. O
papel do paciente é identificado por dois sinais independentes — quem fala mais e quem
*não* abre a consulta — e o código recusa-se a decidir se eles discordarem.

## Controle de custo

Transcribe, Comprehend Medical, Comprehend e Translate cobram por volume. Todo resultado é **cacheado**
(`reports/transcriptions/`, `reports/entities/`, `reports/translations.json`) e nenhum caso
é reprocessado sem `--force`. O modo `--report` recalcula métricas do cache **sem chamar a
AWS**. Os JSON brutos são versionados, permitindo auditar os resultados sem credenciais.
