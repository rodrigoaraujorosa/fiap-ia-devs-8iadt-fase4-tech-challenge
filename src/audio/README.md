# 🎙️ Entrega 2 — Análise de Áudio

Processar áudios de consultas e detectar alterações vocais/respiratórias
(cansaço, dificuldades respiratórias, disartria).

## Pipeline
1. **Azure Speech-to-Text** — transcrição da fala.
2. **Azure Text Analytics** — termos críticos e sentimento sobre a transcrição.
3. **Biomarcadores acústicos** (librosa) — jitter, shimmer, F0, MFCC sobre o áudio bruto.

**Dataset:** Coswara (2.635 indivíduos; respiração, tosse, vogais sustentadas, dígitos
falados + metadados de sintomas). Ver `data/audio/`.

> 🔑 Credenciais Azure via `.env` (modelo em `.env.example`, carregado por
> `src/common/config.py`).
