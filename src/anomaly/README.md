# 📈 Entrega 3 — Detecção de Anomalias

Detectar anomalias em séries temporais de **sinais vitais**, **movimentação do paciente**
e **evolução de prescrições**, gerando alertas para a equipe médica.

Todas as detecções são **não-supervisionadas**; os rótulos do dataset entram apenas na
avaliação, como ground-truth.

## 🧩 Módulos

| Arquivo | Papel |
|---------|-------|
| `cli.py` | **único ponto de entrada**; os demais são bibliotecas sem CLI |
| `movement.py` | UCI HAR — IsolationForest treinado só nas atividades de repouso |
| `vitals.py` | Challenge 2019 — desvio robusto (MAD) contra a linha de base do próprio paciente |
| `prescriptions.py` | variável derivada da FiO2 (dose prescrita) — degrau de 0,15 |
| `report.py` | relatório único para a equipe médica |

## 🚀 Uso

O detector de vitais é **treinado, salvo e depois aplicado** a pacientes que não
participaram do treino:

```bash
# 1. Treina os dois detectores no padrão de normalidade e salva em models/
python -m src.anomaly.cli --train --limit 5000

# 2. Monitoramento de UM indivíduo — é a demonstração do alerta de leito
python -m src.anomaly.cli --monitor p000188        # vitais + prescrições (mesmo paciente)
python -m src.anomaly.cli --monitor-subject 2      # movimentação (outro dataset)

# 3. Avaliação completa + relatório em reports/anomalias.md
python -m src.anomaly.cli --limit 5000
python -m src.anomaly.cli --only movement
```

> ⚠️ **Os dois datasets não têm os mesmos indivíduos.** Não existe o paciente `p000188` no
> UCI HAR. Por isso o monitoramento tem dois comandos: `--monitor` reúne as subtarefas que
> vêm do Challenge 2019 (vitais e dose prescrita, do mesmo paciente) e `--monitor-subject`
> cobre a movimentação, por sujeito.

Roda inteiramente local — não chama a nuvem e não custa nada.

## 🧠 Treino e inferência

```
TREINO (uma vez)   padrão de normalidade ──► models/*.joblib (modelo + limiar)
                   vitais: só pacientes que NUNCA tiveram sepse
                   movimentação: só atividades de repouso

INFERÊNCIA         série de um indivíduo que o modelo não viu
                   └─► instantes em alerta
```

**Exemplo de saída (`--monitor p000188`)** — as subtarefas se complementam:

| Subtarefa | Resultado |
|---|---|
| Sinais vitais | nenhum alerta em 84 horas |
| Prescrições | 2 escalonamentos, horas 51 e 53 |
| Conferência | sepse a partir da hora 75 |

Os vitais ficaram em silêncio; a dose escalonou **24 h antes** do início. É o mesmo efeito
que a comparação vitais × laboratório mostra de forma agregada — os vitais reagem tarde.

A coorte de treino exclui pacientes sépticos **de propósito**: é o padrão de normalidade
que o detector deve aprender. Mantê-los no treino ensinaria ao modelo que a deterioração é
normal. Mesmo desenho da movimentação, onde o modelo só vê repouso.

O limiar é **absoluto** — o percentil 5 dos scores do treino —, não um percentil calculado
dentro de cada paciente. Com corte por paciente, todo paciente recebe alerta por
construção (sempre existe um "5% pior"), o que impede comparar a taxa entre pacientes e
não constitui um detector aplicável.

## 📊 Resultados medidos

| Subtarefa | Ground-truth | Resultado |
|---|---|---|
| Movimentação | atividade real | **F1 0,973 · AUC 0,9999** · recall 100% · falso alarme 5,0% |
| Sinais vitais | `SepsisLabel` | AUC 0,555 · 111/446 avisados na janela de 48 h · lead mediano **30 h** |
| Prescrições | `SepsisLabel` | sepse **17,9%** entre os que escalonaram vs **10,5%** entre os que não |

Vitais e prescrições sobre 5.000 pacientes: 3.187 sem sepse no treino, 1.813 retidos para
teste. As métricas são de **generalização** — séries que o modelo nunca viu.

## ⚠️ Armadilhas já tratadas — não reintroduzir

- **Lead time precisa de janela.** Medir a antecedência a partir do *primeiro alerta da
  internação inteira* infla o número sem significado clínico: há paciente com sepse na
  hora 248 e alertas nas primeiras 60 horas, o que produziria "239 h de antecedência"
  para um alerta sem relação com o evento. Só contam alertas nas **48 h** anteriores ao
  início.
- **Não treinar sobre os dados que se quer avaliar.** A primeira versão fazia
  `model.fit(X)` seguido de `score_samples(X)` sobre o dataset inteiro — isso mede
  memorização, não capacidade de alertar, e não deixa modelo algum para aplicar a um
  paciente novo.
- **Menos da metade dos pacientes com sepse é avisada na janela** (111 de 446). É o preço
  do limiar absoluto, que não garante alerta para todo paciente. Baixá-lo aumenta a
  cobertura ao custo de alarme falso — a escolha depende do ruído que a equipe tolera.
- **`EtCO2` tem 0% de cobertura** no training set A (7,6% no set B) — consta do schema e
  nunca é medida no set A.
  É descartada automaticamente por `live_features()`; sem isso entra no modelo como
  constante zero e dilui a distância entre as amostras.
- **A FiO2 mistura escalas:** parte dos registros usa percentual (21–100) e parte usa
  fração (0,21–1,0). O loader normaliza tudo para fração antes de medir o degrau.
- **Redução de dose não é anomalia.** Só aumentos alertam; reduzir a FiO2 indica que o
  paciente precisa de menos suporte, e marcar isso encheria o alerta de falsos positivos
  benignos.

## 🔍 Onde está o sinal: vitais contra laboratório

Os marcadores de laboratório discriminam sepse **melhor que os sinais vitais**, apesar de
terem cobertura muito menor (4–14% das horas, contra 83–91% dos vitais):

| Features | AUC | AUPRC |
|---|---|---|
| Sinais vitais (7) | 0,571 | 0,029 |
| Marcadores de laboratório (8) | **0,628** | **0,034** |
| Vitais + laboratório (15) | 0,628 | 0,035 |

É coerente com a clínica — lactato e leucócitos são marcadores diretos de sepse, enquanto
a alteração dos vitais é tardia. A entrega mantém os vitais como objeto, conforme o
escopo, e registra a comparação como **limitação medida** (§5 do relatório técnico).

## 🖥️ Por que a detecção é local, e não um serviço gerenciado

As Entregas 1 e 2 usam nuvem onde faz sentido, mas aqui a detecção roda localmente — e a
razão não é preferência técnica: **os dois serviços gerenciados dedicados a anomalia em
séries temporais foram retirados do mercado**. O Azure Anomaly Detector não aceita novos
recursos desde setembro de 2023, e o Amazon Lookout for Metrics encerrou o suporte em 10
de outubro de 2025. A decisão está documentada na §6.2 do relatório técnico.

## 🧪 Testes

```bash
pytest tests/test_anomaly.py
```

18 testes com séries sintéticas — não exigem os datasets baixados. Cobrem sobretudo as
armadilhas acima, que são silenciosas: não levantam erro, só produzem números errados.
