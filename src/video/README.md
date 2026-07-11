# 🎥 Entrega 1 — Análise de Vídeo

Processar vídeos clínicos e detectar movimentos/eventos fora do padrão, gerando
relatórios automáticos de desvios.

## Caminhos
- **Postura (fisioterapia):** OpenPose / MediaPipe Pose sobre o dataset **KIMORE**
  (clinical scores servem de ground-truth para validar desvios posturais).
- **Objetos (cirurgia):** **YOLOv8** para detecção de instrumentos/áreas críticas sobre
  **Cholec80 + Cholec80-Boxes** (15.691 frames com bounding boxes de 7 instrumentos).

> ⚠️ Decisão pendente: seguir OpenPose/KIMORE (postura) **ou** YOLOv8/Cholec80 (instrumentos).
> Datasets diferentes — ver `CLAUDE.md`.

Dados em `data/video/`. Saídas (frames anotados, relatórios) em `reports/`.
