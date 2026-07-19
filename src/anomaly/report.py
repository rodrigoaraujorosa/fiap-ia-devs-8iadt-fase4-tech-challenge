"""
Relatório da Entrega 3 para a equipe médica.

Segue a ordem já usada nas Entregas 1 e 2: gráfico -> análise -> cobertura dos dados ->
estatística -> resumo. Sem ícones, como o relatório técnico.

Módulo de biblioteca: o ponto de entrada é ``python -m src.anomaly.cli``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

FIGURES_DIR = Path("reports/figures")
REPORT_PATH = Path("reports/anomalias.md")


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x:.1%}"


def _num(x: float | None, casas: int = 3) -> str:
    return "—" if x is None else f"{x:.{casas}f}"


def _horas(x: float | None) -> str:
    return "—" if x is None else f"{x:.0f} h"


def _img(path: str | None) -> str | None:
    """Caminho da figura relativo ao diretório do relatório, para o link funcionar."""
    if not path:
        return None
    try:
        return Path(path).resolve().relative_to(REPORT_PATH.resolve().parent).as_posix()
    except ValueError:
        return Path(path).as_posix()


def build_report(vitals: dict, movement: dict, prescriptions: dict,
                 comparison: pd.DataFrame, figures: dict[str, str]) -> str:
    """Monta o markdown do relatório a partir dos resultados das três subtarefas."""
    v, mv, px = vitals["metrics"], movement["metrics"], prescriptions["metrics"]

    L: list[str] = []
    add = L.append

    add("# Entrega 3 — Detecção de Anomalias")
    add("")
    add("Monitoramento de sinais vitais, movimentação e evolução de prescrições em "
        "ambiente de internação. Todas as detecções são **não-supervisionadas** "
        "(IsolationForest ou regra de degrau); os rótulos do dataset entram apenas na "
        "avaliação, como ground-truth.")
    add("")
    add("| Subtarefa | Dataset | Ground-truth | Resultado principal |")
    add("|---|---|---|---|")
    add(f"| Movimentação | UCI HAR | atividade real | F1 {_num(mv['f1'])}, "
        f"AUC {_num(mv['roc_auc'], 4)} |")
    add(f"| Sinais vitais | Challenge 2019 | SepsisLabel | AUC {_num(v['roc_auc'], 4)}, "
        f"lead {_horas(v['lead_median_hours'])} |")
    add(f"| Prescrições (derivada) | Challenge 2019 | SepsisLabel | "
        f"sepse {_pct(px.get('sepsis_rate_escalated'))} vs "
        f"{_pct(px.get('sepsis_rate_not_escalated'))} |")
    add("")

    # ---------------------------------------------------------------- movimentação
    add("## 1. Padrões de movimentação")
    add("")
    if _img(figures.get("movement")):
        add(f"![Alerta por atividade]({_img(figures['movement'])})")
        add("")
    add(f"O modelo é treinado **apenas** com as {mv['train_rest_samples']} amostras de "
        f"repouso (deitado, sentado, em pé) e nunca vê marcha no treino. No teste, com "
        f"{mv['samples']} amostras de {mv['subjects_test']} sujeitos:")
    add("")
    add(f"- **Recall {_pct(mv['recall'])}** — as três atividades de marcha foram "
        f"detectadas integralmente ({mv['tp']} de {mv['tp'] + mv['fn']}).")
    add(f"- **Precisão {_pct(mv['precision'])}**, F1 {_num(mv['f1'])}, "
        f"AUC {_num(mv['roc_auc'], 4)}.")
    add(f"- **Falso alarme {_pct(mv['false_alarm_rate'])}** sobre as amostras de "
        f"repouso ({mv['fp']} de {mv['fp'] + mv['tn']}).")
    add("")
    add("Interpretação clínica: um paciente que deveria estar em repouso e começa a "
        "deambular é detectado sem exceção. O custo é um alarme falso a cada vinte "
        "leituras de repouso, ajustável pelo parâmetro de contaminação.")
    add("")
    add("### Taxa de alerta por atividade real")
    add("")
    add(movement["per_activity"].to_markdown())
    add("")

    # -------------------------------------------------------------- sinais vitais
    add("## 2. Sinais vitais")
    add("")
    if _img(figures.get("vitals")):
        add(f"![Série de vitais com alertas]({_img(figures['vitals'])})")
        add("")
    add(f"Base analisada: **{v['patients']} pacientes**, {v['rows']} horas-paciente, "
        f"{v['sepsis_patients']} com sepse confirmada "
        f"(prevalência horária {_pct(v['prevalence'])}).")
    add("")
    add(f"- AUC {_num(v['roc_auc'], 4)} e AUPRC {_num(v['auprc'], 4)}, contra uma "
        f"prevalência de {_num(v['prevalence'], 4)}.")
    add(f"- **{v['sepsis_patients_alerted']} dos {v['sepsis_patients']}** pacientes com "
        f"sepse receberam algum alerta durante a internação.")
    add(f"- **{v['sepsis_patients_warned']}** foram avisados **dentro da janela de "
        f"{v['lead_window_hours']} h** que antecede o início — antecedência mediana "
        f"**{_horas(v['lead_median_hours'])}**.")
    add("")
    add(f"A distinção entre os dois números acima é deliberada. Contar a antecedência a "
        f"partir do *primeiro alerta da internação inteira* infla o resultado sem "
        f"significado clínico: há paciente com sepse na hora 248 e alertas nas primeiras "
        f"60 horas, o que produziria uma \"antecedência\" de 239 horas para um alerta "
        f"sem relação com o evento. Só contam alertas nas {v['lead_window_hours']} horas "
        f"anteriores ao início.")
    add("")
    add("O resultado hora-a-hora é fraco — a AUC fica pouco acima do acaso. A leitura "
        "honesta é que os sinais vitais isolados não separam bem a hora de sepse da "
        "hora estável.")
    add("")
    add("### Onde está o sinal: vitais contra laboratório")
    add("")
    add(comparison.to_markdown(index=False))
    add("")
    add("Os marcadores de laboratório discriminam melhor que os sinais vitais **apesar "
        "de terem cobertura muito menor** (4% a 14% das horas, contra 83% a 91% dos "
        "vitais). É coerente com a clínica: lactato e leucócitos são marcadores diretos "
        "de sepse, enquanto a alteração dos vitais é tardia. A entrega mantém os vitais "
        "como objeto, conforme o escopo, e registra a comparação como limitação medida.")
    add("")
    add("### Limitações medidas")
    add("")
    add("- O limiar por paciente é percentual: com orçamento de 5% das horas, só rende "
        "um alerta a partir de 20 horas de internação. Os pacientes com sepse que "
        "ficaram sem alerta têm estadias de 8 a 19 horas.")
    add("- `EtCO2` consta do schema mas tem **0% de cobertura** no training set A; a "
        "coluna é descartada automaticamente.")
    add("- O `SepsisLabel` marca a janela em que a sepse é considerada instalada, e não "
        "\"hora anormal\". Um alerta fora dessa janela não é necessariamente falso — "
        "pode ser instabilidade real que não evoluiu para sepse. A precisão hora-a-hora "
        "é, portanto, conservadora por construção.")
    add("")

    # --------------------------------------------------------------- prescrições
    add("## 3. Evolução de prescrições (variável derivada)")
    add("")
    if _img(figures.get("prescriptions")):
        add(f"![Dose prescrita e escalonamentos]({_img(figures['prescriptions'])})")
        add("")
    add("Não existe fonte pública aberta e granular de prescrições hospitalares — a base "
        "de referência é o MIMIC-IV, que exige credenciamento. A subtarefa usa, no lugar, "
        "a **FiO2** (fração inspirada de oxigênio) do próprio Challenge 2019: ao "
        "contrário dos demais campos, que são medições do paciente, a FiO2 é um valor "
        "**prescrito e titulado pela equipe**, e sua série ao longo das horas é uma "
        "série de doses.")
    add("")
    add(f"Com {px['eligible_patients']} pacientes elegíveis "
        f"(de {px['patients_total']}; cobertura da FiO2 {_pct(px['dose_coverage'])}) e "
        f"degrau de {px['threshold']:.2f}:")
    add("")
    add(f"- {px['escalations']} escalonamentos e {px['weanings']} desmames detectados.")
    add(f"- Entre os pacientes que escalonaram, **{_pct(px['sepsis_rate_escalated'])}** "
        f"desenvolveram sepse; entre os que não escalonaram, "
        f"**{_pct(px['sepsis_rate_not_escalated'])}** — cerca de duas vezes mais.")
    add(f"- Recall {_pct(px['recall'])}, precisão {_pct(px['precision'])}.")
    add(f"- **{px['escalated_in_window']}** pacientes escalonaram dentro da janela de "
        f"{px['lead_window_hours']} h antes do início, com antecedência mediana "
        f"**{_horas(px['lead_median_hours'])}**.")
    add("")
    add("Ressalva a declarar: é uma **proxy** de prescrição, não a prescrição registrada "
        "em prontuário. O escalonamento de oxigênio é uma decisão terapêutica real, mas "
        "cobre apenas um eixo do que uma base de prescrições traria.")
    add("")

    # --------------------------------------------------------------------- resumo
    add("## 4. Resumo")
    add("")
    add("As três subtarefas têm desempenhos muito diferentes, e a diferença é "
        "informativa:")
    add("")
    add(f"1. **Movimentação** (AUC {_num(mv['roc_auc'], 4)}) — o problema é quase "
        "separável. Marcha e repouso são estados fisicamente distintos, medidos por "
        "sensores de alta cobertura.")
    add(f"2. **Prescrições** — sinal fraco mas consistente: o escalonamento de dose "
        "dobra a probabilidade de sepse.")
    add(f"3. **Sinais vitais** (AUC {_num(v['roc_auc'], 4)}) — o mais próximo do acaso "
        "hora-a-hora. Deterioração clínica é um processo lento e ruidoso, e os vitais "
        "reagem depois dos marcadores de laboratório.")
    add("")
    add("Para o fluxo de alerta, isso sugere pesos distintos: o alerta de movimentação "
        "pode ser automático; o de vitais deve ser tratado como triagem, para revisão "
        "humana, e não como diagnóstico.")
    add("")
    add("---")
    add("")
    add("Detecção executada localmente com `scikit-learn` (`random_state=42`). Os dois "
        "serviços gerenciados dedicados a anomalia em séries temporais foram retirados "
        "do mercado — o Azure Anomaly Detector não aceita novos recursos desde setembro "
        "de 2023 e o Amazon Lookout for Metrics encerrou o suporte em 10 de outubro de "
        "2025. Ver §6.2 do relatório técnico.")
    add("")
    return "\n".join(L)


def write_report(md: str, path: Path = REPORT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8")
    return path
