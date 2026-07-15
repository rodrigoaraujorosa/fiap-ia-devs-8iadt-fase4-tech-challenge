# 🎥 Entrega 1 — Análise de Vídeo (OpenPose)

Processa vídeos clínicos de reabilitação física (dataset **REHAB24-6** — RGB, com
rótulos de execução correta/incorreta), estima a pose com **OpenPose** e detecta
**desvios posturais fora do padrão**, gerando um relatório automático.

## Arquitetura

O OpenPose (binário externo) só extrai os keypoints; **todo o resto é Python puro**
processando os JSON — o que torna o pipeline portátil e fácil de testar.

```
vídeo .mp4 ──[OpenPose]──► JSON por frame ──► [nosso pipeline] ──► relatório + gráfico
```

| Módulo | Papel |
|--------|-------|
| `run_openpose.py` | invoca o `OpenPoseDemo` sobre um vídeo → pasta de JSONs |
| `keypoints.py` | carrega os JSON BODY_25 → DataFrame (frame × junta x,y,conf); escolhe a pessoa principal e descarta juntas de baixa confiança |
| `posture.py` | ângulos articulares por frame (tronco, quadril, joelho, ombro, cotovelo) |
| `anomaly.py` | desvios = z-score robusto por ângulo **∪** IsolationForest multivariado |
| `report.py` | relatório Markdown + gráfico dos ângulos com anomalias marcadas |
| `overlay.py` | vídeo com o esqueleto sobreposto e os frames de desvio destacados (para o vídeo-demo) |
| `validate.py` | cruza as anomalias com os rótulos `correctness` do REHAB24-6 (via `--segmentation` no CLI) |
| `cli.py` | pipeline fim-a-fim (com barra de progresso no terminal via `tqdm`) |
| `app.py` | app web (Gradio) para o vídeo-demo: progresso, gráfico, relatório e overlay numa tela |

## Uso

```bash
# A partir de JSONs já extraídos
python -m src.video.cli --json-dir reports/json/PM_008 --fps 30

# A partir de um vídeo (roda o OpenPose antes — precisa do binário local)
# --overlay gera o vídeo anotado; --frame-step 3 subamostra (GPU fraca);
# --segmentation ativa a validação contra o ground-truth (correto/incorreto)
python -m src.video.cli --video data/video/rehab24-6/PM_006-Camera17-30fps.mp4 \
    --openpose-root tools/openpose --fps 30 --frame-step 3 --overlay \
    --segmentation data/video/rehab24-6/Segmentation.csv

# Overlay isolado (a partir de JSONs já extraídos + o vídeo)
python -m src.video.overlay --video data/video/rehab24-6/PM_008-Camera17-30fps.mp4 \
    --json-dir reports/json/PM_008 --out reports/PM_008_overlay.mp4
```

Saídas em `reports/`: relatório `.md`, vídeo `_overlay.mp4`, CSV de validação
`validacao_<vídeo>.csv` e o gráfico em `reports/figures/*.png`.

O terminal mostra uma **barra de progresso** do OpenPose (frames prontos/total, contando
os JSONs escritos) e do overlay.

### Interface web (demo)

Para o vídeo-demo, uma app local (Gradio) mostra o progresso e, ao final, o gráfico, o
relatório, a validação e o vídeo overlay tocando — tudo numa tela:

```bash
python -m src.video.app     # abre em http://localhost:7860
```

Escolha o vídeo, ajuste o `frame-step` e clique em **Processar**. Marque *"Reaproveitar
JSONs"* para pular o OpenPose quando já houver keypoints extraídos (demo instantâneo).

## OpenPose

- Setup do binário: [`docs/openpose_setup.md`](../../docs/openpose_setup.md).
- Em GPU fraca, use `--frame-step` para subamostrar e acelerar a extração.

## Testes

```bash
pytest tests/test_video.py -v   # usa keypoints sintéticos, não precisa de OpenPose nem vídeo real
```
