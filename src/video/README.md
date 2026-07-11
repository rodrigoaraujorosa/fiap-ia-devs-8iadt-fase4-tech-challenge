# 🎥 Entrega 1 — Análise de Vídeo (OpenPose)

Processa vídeos clínicos (fisioterapia — dataset **KIMORE**), estima a pose com
**OpenPose** e detecta **desvios posturais fora do padrão**, gerando um relatório
automático.

## Arquitetura

O OpenPose (binário externo) só extrai os keypoints; **todo o resto é Python puro**
processando os JSON — então roda igual na máquina local ou no Google Colab.

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
| `validate.py` | cruza as anomalias com os rótulos `correctness` do REHAB24-6 (validação quantitativa) |
| `cli.py` | pipeline fim-a-fim |

## Uso

```bash
# A partir de JSONs já extraídos (ex.: gerados no Colab)
python -m src.video.cli --json-dir data/video/kimore_ex1_json --fps 30

# A partir de um vídeo (roda o OpenPose antes — precisa do binário local)
# --overlay também gera o vídeo anotado (esqueleto + desvios)
python -m src.video.cli --video data/video/rehab24-6/PM_008.mp4 \
    --openpose-root tools/openpose --fps 30 --overlay

# Overlay isolado (a partir de JSONs já extraídos + o vídeo)
python -m src.video.overlay --video data/video/rehab24-6/PM_008.mp4 \
    --json-dir reports/json/PM_008 --out reports/PM_008_overlay.mp4
```

Saídas em `reports/` (relatório `.md`, vídeo `_overlay.mp4`) e `reports/figures/` (gráfico `.png`).

## OpenPose

- Setup do binário: [`docs/openpose_setup.md`](../../docs/openpose_setup.md).
- Sem GPU boa? Use o notebook [`notebooks/openpose_kimore_colab.ipynb`](../../notebooks/openpose_kimore_colab.ipynb)
  (GPU gratuita do Colab só para extrair os JSONs; a análise roda depois na sua máquina).

## Testes

```bash
pytest tests/test_video.py -v   # usa keypoints sintéticos, não precisa de OpenPose/KIMORE
```
