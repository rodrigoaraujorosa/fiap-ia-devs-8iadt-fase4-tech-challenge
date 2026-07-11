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
