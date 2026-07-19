# Handoff — Tech Challenge Fase 4 (Monitoramento Hospitalar Multimodal)

> Documento para continuar o trabalho em uma nova sessão sem perder contexto.
> Última atualização: 2026-07-18 (fim da sessão da Entrega 2).

## Onde estamos

| Entrega | Estado |
|---|---|
| 1 — Análise de Vídeo (OpenPose) | **Completa e validada** em 3 experimentos (PM_008, PM_034, PM_006) + app Gradio com linguagem para equipe médica + screenshots no relatório |
| 2 — Análise de Áudio (**AWS**) | **Completa e integrada na `main`** (merge `301014f`, 22 commits): Transcribe, Comprehend Medical, Comprehend e Translate; 4 casos processados, WER médio 6,95% |
| 3 — Detecção de Anomalias | **Próxima peça.** Loaders prontos e testados (Challenge 2019 + UCI HAR), mas **nunca executados com dados reais nem documentados** |

## Decisões-chave (não reabrir sem motivo)

- **Dataset de vídeo:** KIMORE ficou indisponível (site fora do ar) e os mirrors só tinham
  esqueleto (sem RGB) → adotamos o **REHAB24-6** (Zenodo, RGB + rótulos correto/incorreto,
  licença CC BY-NC 4.0). DOI 10.5281/zenodo.13305826.
- **Tudo roda LOCAL.** O grupo decidiu não usar Colab, HF Spaces nem Azure para processar
  vídeo (custo/complexidade). Removemos o notebook do Colab. A nuvem **obrigatória** (Azure)
  entra só na **Entrega 2 (áudio)**. O vídeo-demo do desafio mostra a execução local.
- **Detecção de anomalias fica LOCAL** (IsolationForest). Não é preferência técnica: os
  dois serviços gerenciados dedicados foram retirados do mercado — o Azure Anomaly Detector
  não aceita novos recursos desde set/2023, e o **Amazon Lookout for Metrics encerrou o
  suporte em 10/out/2025**. Verificado; está na §6.2 do relatório técnico.
- **OpenPose = binário externo** (v1.7.0 portable), não instala via pip. Chamado por
  subprocess; gera **1 JSON por frame**.

## Ambiente

- Windows 11, **Python 3.12.6**, GPU **NVIDIA MX330 (2 GB)** — fraca; OpenPose ~1,2 s/frame.
- ⚠️ **O projeto roda em `.venv` (gitignored, na raiz). SEMPRE ative antes de rodar
  qualquer comando** — `.venv\Scripts\Activate.ps1` no PowerShell. A máquina também tem um
  Python de sistema em `C:\Python312` com **versões diferentes** das bibliotecas
  (scikit-learn 1.7.2 no sistema vs. 1.9.0 no venv, pandas 2.3.3 vs. 3.0.3). Rodar no
  interpretador errado não quebra, mas mede outro ambiente. Confira com
  `python -c "import sys; print(sys.prefix)"`.
- OpenPose em `tools/openpose/` (gitignored). Modelo BODY_25 baixado de mirror HuggingFace
  (o servidor da CMU está morto) — ver `docs/openpose_setup.md`.
- Dependências em `requirements.txt` (pisos flexíveis) e **`requirements-lock.txt`**
  (versões exatas do `.venv`, para auditar os números do relatório).
- **Nuvem: AWS**, região `us-east-1`. A Azure foi descartada por falta de cota no Azure
  for Students; a AWS foi liberada depois. Credenciais em `~/.aws/credentials` (via
  `aws configure`); o `.env` guarda **apenas** região e bucket, sem segredos. Bucket:
  `amzn-s3-fase4-techchallenge-...` (nome completo no `.env` — o console da AWS acrescenta
  sufixo com id da conta). Verificar o ambiente com `python -m src.common.config`.

## Pipeline de Vídeo — `src/video/`

| Módulo | Papel |
|---|---|
| `run_openpose.py` | chama o binário; progresso contando os JSONs escritos; saída em log |
| `keypoints.py` | parser BODY_25 + **seleção robusta de pessoa** (maior bbox × confiança + estabilização temporal — resolve "pessoa ao fundo") |
| `posture.py` | 9 ângulos (tronco, quadris, joelhos, ombros, cotovelos) |
| `anomaly.py` | desvios = z-score robusto (MAD) ∪ IsolationForest |
| `report.py` | relatório md: **gráfico → análise → cobertura → estatística → resumo/eventos**; inclui exercício e tempo de processamento |
| `overlay.py` | vídeo anotado; transcodado p/ **H.264** (imageio-ffmpeg) p/ tocar no navegador |
| `validate.py` | cruza com `Segmentation.csv` (correto/incorreto) + `exercise_label` |
| `cli.py` | pipeline fim-a-fim + barra `tqdm` |
| `app.py` | app Gradio (`python -m src.video.app`, porta 7860); linguagem para equipe médica; `show_progress_on=[graph, overlay]` evita barras flutuando sobre os Markdown (**exige gradio >= 5.0**) |

## Pipeline de Áudio — `src/audio/` (Entrega 2, concluída)

```
consulta (.mp3) ──► S3 ──► Transcribe ──► diarização ──► Comprehend Medical ──► relatório
                                                      └─► Comprehend (sentimento)  bilíngue
```

| Módulo | Papel |
|---|---|
| `cli.py` | **único ponto de entrada**; os demais são bibliotecas sem CLI |
| `consultations.py` | loader do dataset; separa turnos `D:`/`P:` e isola a fala do paciente |
| `transcribe.py` | upload S3 → job do Transcribe → diarização → WER contra a referência humana |
| `comprehend.py` | entidades clínicas (Medical) + sentimento (Comprehend geral) |
| `report.py` | relatório bilíngue para a equipe médica (Translate) |
| `cache.py` | cache em disco dos resultados pagos — evita cobrança repetida |

**Dataset:** consultas médicas simuladas (figshare, DOI 10.6084/m9.figshare.16550013.v1,
CC0). 272 casos, 213 respiratórios. Em `data/audio/consultas/` (2 GB, gitignored).

**Resultados (4 casos):** WER médio **6,95%** (4,12% a 10,79%); recall de achados clínicos
**0,806**; sentimento NEGATIVE nos quatro.

**Armadilhas já tratadas — não reintroduzir:**
- 2 dos 213 casos respiratórios estão em **UTF-16**; ler tudo como UTF-8 não levanta erro,
  devolve texto corrompido e o caso aparece com zero turnos.
- Só a **fala do paciente** vai para o Comprehend Medical; as perguntas do médico contêm
  termos clínicos que não são achados dele.
- O **limiar de 0,70** em `clinical_findings()` cria efeito de degrau: 4 dos 7 achados
  "não recuperados" estão presentes na extração, apenas abaixo do corte.
- O serviço extrai como achado expressões de **ausência** de problema ("head was fine",
  "throat felt ok."); o traço `NEGATION` não pega, porque a frase é afirmativa.
- Truncar trecho no meio da palavra corrompe a tradução ("on my che" virou "na minha
  cabeça" em vez de "no meu peito") — usar `_shorten()`.

**Custo:** tudo cacheado em `reports/transcriptions/`, `reports/entities/` e
`reports/translations.json`. Nenhum caso reprocessa sem `--force`. Use `--dry-run` para ver
o que seria cobrado e `--report` para recalcular métricas sem tocar na AWS.

## Comandos

```bash
# 0. SEMPRE primeiro: ativar o venv
.venv\Scripts\Activate.ps1         # PowerShell
# source .venv/Scripts/activate    # Git Bash

# --- Entrega 1 (vídeo) ---
# CLI completo (extrai + analisa + overlay + valida), com progresso
python -m src.video.cli --video data/video/rehab24-6/PM_034-Camera17-30fps.mp4 \
    --openpose-root tools/openpose --fps 30 --frame-step 3 --overlay \
    --segmentation data/video/rehab24-6/Segmentation.csv

# App web de demonstração
python -m src.video.app            # http://localhost:7860

# --- Entrega 2 (áudio) ---
# Consultas médicas: estatísticas e falas do paciente
python -m src.audio.consultations --root data/audio/consultas --summary
python -m src.audio.consultations --root data/audio/consultas --case RES0001 --patient-only

# Pipeline completo (CUSTA DINHEIRO — cacheia tudo em reports/)
python -m src.audio.cli --case RES0091
python -m src.audio.cli --case RES0091 --dry-run   # o que seria cobrado, sem executar

# Etapas isoladas, para depurar
python -m src.audio.transcribe --report            # métricas só do cache
python -m src.audio.comprehend --cases RES0029 --compare
python -m src.audio.comprehend --cases RES0029 --sentiment
python -m src.audio.report --case RES0029

# Verificação do ambiente AWS (não custa nada)
python -m src.common.config

# Testes (9 passando)
pytest
```

## Dados (`data/video/rehab24-6/`, gitignored)

- `videos.zip` (2,7 GB), `Segmentation.csv` (rótulos), + vídeos extraídos:
  - **PM_034** — Ex4 **abdução de perna**, ~37 s (1.108 frames), 10 repetições (5 corretas,
    5 incorretas), **sem pessoas ao fundo**. **É o vídeo da demonstração.** Validado:
    33,0% de frames com desvio; taxa média **correto 0,034 vs incorreto 0,358** —
    separação ~10x, bem mais nítida que a do PM_008.
  - **PM_008** — Ex6 **agachamento**, ~3 min (5.191 frames), 27 repetições. Limpo
    (1 paciente). Caso de **validação quantitativa** mais extenso: 33,3% de frames com
    desvio; taxa média **correto 0,430 vs incorreto 0,614** (separação ~1,4x — o agachamento
    é de grande amplitude, então até a execução correta se afasta da mediana do vídeo).
  - **PM_006** — Ex4, ~35 s, 10 repetições (5/5). Caso de **condições adversas de
    captura**: luzes apagadas, câmera em meio-perfil e 1 pessoa ao fundo. Taxa média
    **correto 0,175 vs incorreto 0,488** (razão 2,8x, contra 10,5x do PM_034) — a separação
    resiste ao pior cenário. **Não é experimento controlado:** difere do PM_034 também no
    sujeito (person_id 1 vs 4) e na perna (direita vs esquerda), então a queda não é
    atribuível a um fator isolado. Ver a ressalva na §3.3 do relatório técnico.

## Workflow do usuário (importante)

- Ele roda o CLI/app e depois pede para **"apagar todos os artefatos gerados"**: apagar
  `reports/json/`, `reports/video/`, `reports/*_overlay.mp4`, `reports/relatorio_*.md`,
  `reports/validacao_*.csv`, `reports/figures/angulos_*.png` — **mantendo** `videos.zip`,
  `Segmentation.csv` e os vídeos-fonte extraídos.
- `reports/json/` é gitignored (centenas de JSONs). Gráficos de relatório (`reports/figures/*.png`)
  são versionáveis (exceção no `.gitignore`).

## Convenções de código (importante)

- **Código em inglês, comentários em pt-BR.** Nomes de arquivo, funções, variáveis, colunas
  de DataFrame e **parâmetros de CLI** em inglês (`--frame-step`, `--per-group`,
  `--keep-fillers`); docstrings e comentários em português. A Entrega 1 já seguia isso; a
  Entrega 2 nasceu em português e foi convertida — não reintroduzir.
- **Relatórios para a equipe médica são bilíngues.** O áudio-fonte é em inglês, então o
  trecho citado (transcrição, entidade extraída) aparece **no original em inglês, seguido
  da tradução em pt-BR**. Nunca só em inglês, nunca traduzido sem o original — a equipe
  precisa poder conferir contra a gravação.

## Preferências

- **README** com emojis/badges (default do usuário). **Relatório técnico SEM ícones.**
- **Grupo 30**, turma 8IADT. Integrantes: Rodrigo de Araújo Rosa, Elias Maximiano da Silva,
  Fábia Gomes de Jesus, Danilo Pereira.
- `CLAUDE.md` é gitignored — **manter como está** (não editar).
- Commits frequentes, mensagens em PT-BR, terminando com `Co-Authored-By: Claude ...`.

## Próximos passos

1. ~~Entrega 1 (vídeo)~~ — **concluída.**
2. ~~Entrega 2 (áudio)~~ — **concluída e integrada na `main`** (merge `301014f`).
3. **➡️ ENTREGA 3 — é por aqui que a próxima sessão começa.** Ver o roteiro abaixo.
4. **Relatório técnico**: completar as seções **[Em desenvolvimento]** — 5.4 (prescrições),
   6.4 (fluxo de alerta) e 12 (conclusão).
5. **Vídeo demo** (YouTube/Vimeo, até 15 min). Orçamento acordado: 1min abertura, 3min30
   vídeo, 4min áudio, 2min30 anomalias, 2min nuvem/alerta, 1min conclusão, 1min folga.

## Roteiro da Entrega 3 (próxima sessão)

**O que já existe:** `src/anomaly/load_challenge2019.py` e `load_uci_har.py` — loaders com
baseline IsolationForest, **testados com dados sintéticos mas nunca executados com os
dados reais**. Nada foi documentado no relatório além do plano (§5).

**Passos sugeridos:**

1. Baixar os datasets (instruções em `docs/datasets_README.md`, seções 2 e 3):
   PhysioNet Challenge 2019 (~42 MB) e UCI HAR (~60 MB). Cabem tranquilamente — mas
   **atenção ao espaço em disco**, que estava em 28 GB livres de 459 GB.
2. Rodar os dois loaders e conferir se funcionam com os dados reais (foram escritos há
   várias sessões, contra dados sintéticos).
3. Decidir a subtarefa de **prescrições**: Synthea (sintético) ou variável derivada do
   Challenge 2019. Está em aberto desde o início.
4. Documentar na §5 do relatório técnico, seguindo o padrão das Entregas 1 e 2:
   resultados medidos, figuras numeradas **a partir da 9**, e as limitações encontradas.

**Padrão a seguir, já estabelecido nas duas primeiras entregas:**
- validar contra o ground-truth disponível (`SepsisLabel` no Challenge 2019; os rótulos de
  atividade no UCI HAR) em vez de só exibir o resultado;
- `random_state=42` fixo, para reprodutibilidade;
- CLI como único ponto de entrada, se for criar um;
- código em inglês, comentários em pt-BR.

## Pendências de fechamento (fazer ao final, com tudo pronto)

- [ ] **Glossário no relatório técnico.** Decisão do usuário: fazer no fim, quando todas as
      entregas estiverem escritas, para o glossário nascer completo. Levantamento das siglas
      que aparecem 2+ vezes e **nunca são explicadas** no texto:

      WER, REHAB24-6, CLI, HAR, VRAM, CUDA, MAD, UTI, UTF-8/UTF-16,
      NEGATION / NEGATIVE (traços e rótulos dos serviços AWS)

      O **WER** é o mais crítico: aparece 10 vezes, é a métrica central da Entrega 2 e nunca
      é expandido. Definição: *Word Error Rate*, taxa de erro de palavra —
      (substituições + inserções + deleções) / palavras da referência.

- [ ] **Poda do README raiz.** A seção "Configuração da AWS" ficou detalhada demais para um
      README; parte dela cabe melhor no relatório técnico ou em `docs/`.

- [ ] Revisar se os relatórios de saída em `reports/audio_*.md` devem ficar versionados ou
      se apenas um exemplo basta.
