# Handoff — Tech Challenge Fase 4 (Monitoramento Hospitalar Multimodal)

> Documento para continuar o trabalho em uma nova sessão sem perder contexto.
> Última atualização: 2026-07-18.

## Onde estamos

| Entrega | Estado |
|---|---|
| 1 — Análise de Vídeo (OpenPose) | **Completa e validada** em 3 experimentos (PM_008, PM_034, PM_006) + app Gradio com linguagem para equipe médica + screenshots no relatório |
| 2 — Análise de Áudio (**AWS**) | **Em andamento** na branch `feature/entrega-2-audio`. Dois datasets baixados e com loader validado; falta Transcribe, Comprehend Medical e biomarcadores |
| 3 — Detecção de Anomalias | **Baseline pronto** (loaders Challenge 2019 + UCI HAR); falta rodar/documentar |

## Decisões-chave (não reabrir sem motivo)

- **Dataset de vídeo:** KIMORE ficou indisponível (site fora do ar) e os mirrors só tinham
  esqueleto (sem RGB) → adotamos o **REHAB24-6** (Zenodo, RGB + rótulos correto/incorreto,
  licença CC BY-NC 4.0). DOI 10.5281/zenodo.13305826.
- **Tudo roda LOCAL.** O grupo decidiu não usar Colab, HF Spaces nem Azure para processar
  vídeo (custo/complexidade). Removemos o notebook do Colab. A nuvem **obrigatória** (Azure)
  entra só na **Entrega 2 (áudio)**. O vídeo-demo do desafio mostra a execução local.
- **Azure Anomaly Detector** está aposentado (não permite criar recurso novo desde 2023) →
  Entrega 3 fica com detecção **local** (IsolationForest), documentado.
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
- **Nuvem: AWS** (Transcribe + Comprehend Medical). A Azure foi descartada — usuário tem
  e-mail acadêmico mas SEM cota no Azure for Students; a AWS foi liberada depois.

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
# Coswara: estatísticas e coorte equilibrada
python -m src.audio.dataset --root data/audio/coswara --resumo
python -m src.audio.dataset --root data/audio/coswara --coorte \
    --lotes 20220224 20210406 --por-grupo 30

# Consultas médicas: estatísticas e recorte
python -m src.audio.consultas --root data/audio/consultas --resumo
python -m src.audio.consultas --root data/audio/consultas --caso RES0001 --paciente

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

## Preferências

- **README** com emojis/badges (default do usuário). **Relatório técnico SEM ícones.**
- **Grupo 30**, turma 8IADT. Integrantes: Rodrigo de Araújo Rosa, Elias Maximiano da Silva,
  Fábia Gomes de Jesus, Danilo Pereira.
- `CLAUDE.md` é gitignored — **manter como está** (não editar).
- Commits frequentes, mensagens em PT-BR, terminando com `Co-Authored-By: Claude ...`.

## Entrega 2 — Áudio (`src/audio/`, branch `feature/entrega-2-audio`)

**Duas fontes, por necessidade técnica** — não é redundância:

| Dataset | O que fornece | Alimenta |
|---|---|---|
| **Coswara** (`dataset.py`) | fonação sustentada, respiração e tosse, com sintoma por participante | biomarcadores (librosa) |
| **Consultas simuladas** (`consultas.py`) | fala clínica espontânea + transcrição humana | Transcribe → Comprehend Medical |

Jitter/shimmer/F0 exigem **vogal sustentada** — não se calcula de forma confiável em
conversa espontânea com dois falantes. E o Coswara só tem gente **contando números**, que
não gera linguagem clínica para o Comprehend Medical extrair. Nenhum substitui o outro.

- Coswara: 2 lotes baixados (20220224 e 20210406, 1,7 GB). Coorte equilibrada 30+30.
  **Controle exige ausência de sintomas**, não só `covid_status == healthy` (121 dos 1.433
  "healthy" relatam sintoma).
- Consultas: 272 casos (213 respiratórios), CC0, figshare DOI 10.6084/m9.figshare.16550013.
  As transcrições humanas são **ground-truth para medir o WER do Transcribe**.
  Armadilha tratada: 2 arquivos estão em UTF-16, o resto em UTF-8.

## Próximos passos

1. ~~Vídeo curto sem pessoa ao fundo para o demo~~ — **resolvido: PM_034.** Entrega 1 fechada.
2. **Entrega 2:** `pip install boto3` **no venv** (pendente); criar bucket S3 e definir a
   região (`us-east-1` sugerida — confirmar disponibilidade do Comprehend Medical); depois
   escrever upload S3 → Transcribe → Comprehend Medical e os biomarcadores.
3. **Entrega 3:** baixar Challenge 2019 + UCI HAR, rodar e documentar; Synthea p/ prescrições.
4. **Relatório técnico**: completar as seções **[Em desenvolvimento]**. Atenção: as seções
   **4 e 6 ainda descrevem o pipeline Azure** — reescrever para AWS quando a Entrega 2
   estiver rodando, com resultados reais em vez de promessas.
5. **Vídeo demo** (YouTube/Vimeo, até 15 min). Orçamento acordado: 1min abertura, 3min30
   vídeo, 4min áudio, 2min30 anomalias, 2min nuvem/alerta, 1min conclusão, 1min folga.
