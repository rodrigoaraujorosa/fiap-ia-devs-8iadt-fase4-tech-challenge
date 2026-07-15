# Handoff — Tech Challenge Fase 4 (Monitoramento Hospitalar Multimodal)

> Documento para continuar o trabalho em uma nova sessão sem perder contexto.
> Última atualização: 2026-07-14.

## Onde estamos

| Entrega | Estado |
|---|---|
| 1 — Análise de Vídeo (OpenPose) | **Completa e validada** + app Gradio de demo + barra de progresso |
| 2 — Análise de Áudio (Azure) | **Não iniciada** (próxima grande peça) |
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
- OpenPose em `tools/openpose/` (gitignored). Modelo BODY_25 baixado de mirror HuggingFace
  (o servidor da CMU está morto) — ver `docs/openpose_setup.md`.
- Dependências em `requirements.txt`. Recentes: `gradio`, `imageio-ffmpeg` (ffmpeg embutido
  p/ transcodar o overlay a H.264).
- Usuário tem e-mail acadêmico, **mas SEM cota no Azure for Students**.

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
| `app.py` | app Gradio (`python -m src.video.app`, porta 7860) |

## Comandos

```bash
# CLI completo (extrai + analisa + overlay + valida), com progresso
python -m src.video.cli --video data/video/rehab24-6/PM_008-Camera17-30fps.mp4 \
    --openpose-root tools/openpose --fps 30 --frame-step 3 --overlay \
    --segmentation data/video/rehab24-6/Segmentation.csv

# App web de demonstração
python -m src.video.app            # http://localhost:7860

# Testes (9 passando)
pytest
```

## Dados (`data/video/rehab24-6/`, gitignored)

- `videos.zip` (2,7 GB), `Segmentation.csv` (rótulos), + vídeos extraídos:
  - **PM_008** — Ex6 **agachamento**, ~3 min (5.191 frames). Limpo (1 paciente). Validado:
    33,3% de frames com desvio; taxa média **correto 0,430 vs incorreto 0,614** (incorretos
    concentram mais desvios — separação OK).
  - **PM_006** — Ex4 **abdução de perna**, ~31 s. **Tem uma pessoa ao fundo** (fica estranho
    no vídeo). Usuário vai procurar um vídeo **curto e sem gente ao fundo** para substituir.

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

## Próximos passos

1. Usuário busca um vídeo curto **sem pessoa ao fundo** (melhor que o PM_006) para o demo.
2. **Entrega 2 (Áudio/Azure):** baixar **Coswara**; pipeline Azure Speech-to-Text + Text
   Analytics (Health) + biomarcadores acústicos (librosa). Credenciais Azure via `.env`
   (modelo em `.env.example`). Estruturar `src/audio/`.
3. **Entrega 3:** baixar Challenge 2019 + UCI HAR, rodar e documentar; Synthea p/ prescrições.
4. **Relatório técnico** (`reports/TECHNICAL_REPORT_FASE4.md`): completar as seções
   **[Em desenvolvimento]** (áudio, integração/alerta em nuvem, conclusão).
5. **Vídeo demo** (YouTube/Vimeo, até 15 min) mostrando o fluxo multimodal.
