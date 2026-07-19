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


def plot_monitor(df: pd.DataFrame, patient: str, out_path: str,
                 window: int = 48) -> str | None:
    """
    Figura do monitoramento de um paciente: sinais vitais e dose na mesma linha do tempo.

    Dois painéis empilhados que **compartilham o eixo x**, porque a pergunta que a figura
    responde é de posição: onde os alertas caem em relação ao início da sepse. Painéis
    separados, com escalas de tempo independentes, não responderiam isso.

    O ``df`` esperado é a série já pontuada pelo detector de vitais e passada pela regra
    de dose — tem ``is_anomaly``, ``dose``, ``is_escalation`` e ``SepsisLabel``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = df[df["patient"] == patient] if "patient" in df.columns else df
    if sub.empty:
        return None

    tem_dose = sub["dose"].notna().any() if "dose" in sub.columns else False
    fig, eixos = plt.subplots(
        2 if tem_dose else 1, 1, figsize=(11, 6.5 if tem_dose else 4.2),
        sharex=True, gridspec_kw={"height_ratios": [2, 1]} if tem_dose else None)
    eixos = eixos if tem_dose else [eixos]
    ax_v = eixos[0]

    # --- janela de aviso e início da sepse, desenhados antes para ficarem no fundo
    onset = None
    if sub["SepsisLabel"].max() == 1:
        onset = int(sub.loc[sub["SepsisLabel"] == 1, "hour"].min())
        for ax in eixos:
            ax.axvspan(max(0, onset - window), onset, color="#f39c12", alpha=0.12,
                       label=f"janela de {window} h" if ax is ax_v else None)
            ax.axvline(onset, color="#8e44ad", linestyle="--", linewidth=1.8,
                       label="início da sepse" if ax is ax_v else None)

    # --- painel 1: sinais vitais
    for v, cor in [("HR", "#2e86c1"), ("SBP", "#e67e22"),
                   ("Resp", "#27ae60"), ("O2Sat", "#c0392b")]:
        ax_v.plot(sub["hour"], sub[v], label=v, alpha=0.85, linewidth=1.2, color=cor)

    # Horas consecutivas viram uma faixa. Desenhar uma linha por hora funciona quando há
    # poucos alertas, mas um paciente com 197 horas sinalizadas em 258 vira um bloco
    # vermelho sólido — a figura deixa de informar onde os alertas estão.
    horas_alerta = sorted(sub.loc[sub["is_anomaly"] == 1, "hour"].tolist())
    faixas: list[tuple[int, int]] = []
    for h in horas_alerta:
        if faixas and h == faixas[-1][1] + 1:
            faixas[-1] = (faixas[-1][0], h)
        else:
            faixas.append((h, h))
    for i, (ini, fim) in enumerate(faixas):
        ax_v.axvspan(ini - 0.5, fim + 0.5, color="#c0392b", alpha=0.30,
                     label="horas em alerta" if i == 0 else None)

    ax_v.set_ylabel("valor medido")
    titulo = f"Monitoramento do paciente {patient}"
    if onset is not None:
        titulo += f" — sepse na hora {onset}"
    ax_v.set_title(titulo)
    ax_v.legend(loc="upper left", fontsize=8, ncol=2)
    if not tem_dose:
        ax_v.set_xlabel("hora de internação")

    # --- painel 2: dose prescrita
    if tem_dose:
        ax_d = eixos[1]
        com_dose = sub.dropna(subset=["dose"])
        ax_d.step(com_dose["hour"], com_dose["dose"], where="post",
                  color="#2e86c1", linewidth=2, label="FiO2 — dose prescrita")
        esc = sub[sub["is_escalation"] == 1]
        if len(esc):
            ax_d.scatter(esc["hour"], esc["dose"], color="#c0392b", zorder=5, s=70,
                         marker="^", label="escalonamento de dose")
        ax_d.set_xlabel("hora de internação")
        ax_d.set_ylabel("FiO2")
        ax_d.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    caminho = Path(out_path)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    return str(caminho)


def plot_subject_timeline(res: pd.DataFrame, subject: int, out_path: str) -> str | None:
    """
    Figura do monitoramento de movimentação: atividade real e alerta ao longo do tempo.

    A taxa de alerta por atividade (``movement.plot``) mostra o agregado, mas não deixa
    ver o comportamento na sequência. Aqui cada janela de leitura vira uma coluna: em
    cima a atividade que o sujeito realizava, embaixo se o alerta disparou. Fica visível
    de relance que os blocos de marcha acendem por inteiro e os de repouso ficam quase
    todos apagados.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if res.empty:
        return None

    from .movement import REST_ACTIVITIES

    ordem = ["LAYING", "SITTING", "STANDING",
             "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"]
    presentes = [a for a in ordem if a in set(res["activity"])]
    pos = {a: i for i, a in enumerate(presentes)}

    x = range(len(res))
    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(11, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    # painel de cima: atividade real, colorida por classe
    for i, (_, r) in enumerate(res.iterrows()):
        movimento = r["activity"] not in REST_ACTIVITIES
        ax_a.bar(i, 1, bottom=pos[r["activity"]], width=1.0,
                 color="#c0392b" if movimento else "#2e86c1", linewidth=0)

    ax_a.set_yticks([v + 0.5 for v in pos.values()])
    ax_a.set_yticklabels(presentes, fontsize=8)
    ax_a.set_ylim(0, len(presentes))
    ax_a.set_title(f"Movimentação do sujeito {subject} — atividade real e alerta")
    # legenda fora da área de dados: dentro, ela cobre os blocos do canto superior
    ax_a.legend(handles=[Patch(color="#2e86c1", label="repouso (esperado em leito)"),
                         Patch(color="#c0392b", label="marcha (deve alertar)")],
                loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=2, fontsize=8,
                frameon=False)

    # painel de baixo: o que o detector disparou
    ax_b.bar(x, res["is_anomaly"], width=1.0, color="#c0392b", linewidth=0)
    ax_b.set_yticks([0, 1])
    ax_b.set_yticklabels(["sem alerta", "ALERTA"], fontsize=8)
    ax_b.set_ylim(-0.1, 1.1)
    ax_b.set_xlabel("janela de leitura (ordem de aquisição)")

    fig.tight_layout()
    caminho = Path(out_path)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    return str(caminho)


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
    add(f"O detector é **treinado em {v['train_patients']} pacientes que nunca "
        f"desenvolveram sepse** ({v['train_hours']} horas) e avaliado em "
        f"**{v['patients']} pacientes retidos**, que não aparecem no treino — "
        f"{v['rows']} horas, {v['sepsis_patients']} com sepse "
        f"(prevalência horária {_pct(v['prevalence'])}).")
    add("")
    add("A coorte de treino exclui pacientes sépticos de propósito: é o padrão de "
        "normalidade que o detector deve aprender. Mantê-los no treino ensinaria ao "
        "modelo que a deterioração é normal. As métricas abaixo são, portanto, de "
        "**generalização** — o que o detector faz com séries que nunca viu.")
    add("")
    add(f"O limiar de alerta é **absoluto** ({v['threshold']:.4f}), fixado no percentil "
        f"{v['contamination']:.0%} dos scores do treino, e não um percentil calculado "
        f"dentro de cada paciente. A diferença é operacional: com corte percentual por "
        f"paciente, todo paciente recebe alerta por construção — sempre existe um "
        f"\"5% pior\" — e a taxa deixa de ser comparável entre pacientes.")
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
    add("- **Menos da metade dos pacientes com sepse é avisada dentro da janela.** É a "
        "consequência direta de usar um limiar absoluto: ele não garante alerta para "
        "todo paciente, ao contrário do corte percentual por paciente, que garantia mas "
        "não era um detector aplicável. Baixar o limiar aumenta a cobertura ao custo de "
        "mais alarme falso — a escolha depende de quanto ruído a equipe tolera.")
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
    add(f"- {px['escalations']} escalonamentos e {px['reductions']} reduções de dose.")
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
