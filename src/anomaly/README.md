# 📈 Entrega 3 — Detecção de Anomalias

Detectar anomalias em séries temporais de sinais vitais, evolução de prescrições e
padrões de movimentação do paciente, gerando alertas automáticos.

## Módulos
| Arquivo | Subtarefa | Dataset | Baseline |
|---------|-----------|---------|----------|
| `load_challenge2019.py` | Sinais vitais (UTI, horários) | PhysioNet Challenge 2019 | IsolationForest sobre 8 vitais |
| `load_uci_har.py` | Movimentação do paciente | UCI HAR | IsolationForest treinado só em repouso |
| _(a fazer)_ | Evolução de prescrições | Synthea (sintético) | detecção de mudança brusca |

## Uso
```bash
python src/anomaly/load_challenge2019.py --data ./data/anomaly/challenge2019 --patient p000001
python src/anomaly/load_uci_har.py --data "./data/anomaly/uci_har/UCI HAR Dataset"
```

`SepsisLabel` (Challenge 2019) e as atividades de movimento (UCI HAR) servem de
ground-truth para validar as anomalias detectadas.

> ⚠️ **Estado:** os loaders foram escritos e testados com dados sintéticos, mas **ainda não
> foram executados com os datasets reais** nem documentados no relatório técnico. É a
> próxima peça do trabalho.

## Por que a detecção é local, e não um serviço gerenciado

As Entregas 1 e 2 usam nuvem onde faz sentido, mas aqui a detecção roda localmente — e a
razão não é preferência técnica: **os dois serviços gerenciados dedicados a anomalia em
séries temporais foram retirados do mercado**. O Azure Anomaly Detector não aceita novos
recursos desde setembro de 2023, e o Amazon Lookout for Metrics encerrou o suporte em 10 de
outubro de 2025. A decisão está documentada na §6.2 do relatório técnico.
