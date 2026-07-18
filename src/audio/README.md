# 🎙️ Entrega 2 — Análise de Áudio (AWS)

Processar áudios de participantes e detectar alterações vocais/respiratórias
(cansaço, dificuldades respiratórias), transcrever a fala e extrair entidades clínicas.

## Pipeline

1. **Amazon Transcribe** — transcrição da fala.
2. **Amazon Comprehend Medical** — entidades clínicas (sintomas, medicações, anatomia)
   sobre a transcrição.
3. **Biomarcadores acústicos** (librosa) — jitter, shimmer, F0, MFCC sobre o áudio bruto.

> **Provedor de nuvem: AWS.** O plano original usava Azure Cognitive Services, mas não
> havia cota disponível. Com a liberação da AWS, o pipeline passou a usar Transcribe +
> Comprehend Medical — este último é um encaixe melhor que o equivalente da Azure para o
> objetivo da entrega, por extrair entidades clínicas já tipadas.

🔑 Credenciais via `.env` (modelo em `.env.example`) ou `~/.aws/credentials`, carregadas por
`src/common/config.py`. O Transcribe **exige que o áudio esteja no S3** — não aceita upload
direto na chamada.

## Dataset: Coswara

Áudios de respiração, tosse, vogais sustentadas e contagem de números, com metadados de
sintomas e comorbidades. Cada participante contribui com **nove** gravações.

| Item | Valor |
|---|---|
| Fonte | [github.com/iiscleap/Coswara-Data](https://github.com/iiscleap/Coswara-Data) |
| Participantes | 2.746 (`combined_data.csv`) |
| Tamanho total | ~28 GB (45 lotes) — baixamos apenas 2 lotes, ~1,7 GB |
| Licença | Open-access, não-comercial |

O dataset traz **rótulos de qualidade por gravação** (`annotations/`), avaliados por escuta
manual: 0 ruim, 1 boa, 2 excelente. Filtrar por eles evita mandar áudio inaudível para a
nuvem — o que custa dinheiro e polui o resultado.

### Lotes usados

| Lote | Tamanho | Papel |
|---|---|---|
| `20220224` | 1.369 MB | rico em sintomáticos (114 com áudio de fala bom) |
| `20210406` | 317 MB | reforça o grupo de controle (mais 40 saudáveis) |

## Uso

```bash
# estatísticas dos metadados (não precisa do áudio)
python -m src.audio.dataset --root data/audio/coswara --resumo

# monta a coorte equilibrada e salva em CSV
python -m src.audio.dataset --root data/audio/coswara --coorte \
    --lotes 20220224 20210406 --por-grupo 30 --out reports/coorte_audio.csv

# junta as partes .tar.gz.a* e extrai um lote
python -m src.audio.dataset --root data/audio/coswara --extrair 20220224
```

## Definição dos grupos

- **sintomático** — relata `bd` (dificuldade respiratória) **ou** `ftg` (fadiga), que são
  exatamente os sintomas do enunciado do desafio.
- **saudável** — `covid_status == healthy` **e nenhum sintoma relatado**.

A segunda condição não é redundante: **121 dos 1.433 participantes declarados `healthy`
relatam algum sintoma**, sendo 12 com dificuldade respiratória e 12 com fadiga. Usar apenas
o `covid_status` contaminaria o grupo de controle com casos que pertencem ao outro grupo.

## Estrutura dos dados (`data/audio/coswara/`, gitignored)

```
combined_data.csv          metadados de todos os participantes
csv_labels_legend.json     legenda das colunas abreviadas (bd, ftg, ...)
folder_csv/<lote>.csv      mapa participante -> lote (necessário: record_date não serve)
annotations/*_labels.csv   qualidade por gravação
raw/<lote>/*.tar.gz.a*     partes baixadas
extracted/<lote>/<id>/     áudio extraído, um .wav por som
```
