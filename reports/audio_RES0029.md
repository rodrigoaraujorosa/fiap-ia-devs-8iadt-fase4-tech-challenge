# Relatório de análise de áudio — consulta RES0029

*Gerado em 18/07/2026 às 21:36. Documento de apoio: não substitui a avaliação clínica.*

O áudio original está em inglês. Cada termo e cada trecho aparecem no **original**, seguidos da **tradução para o português**.

## Achados relatados pelo paciente

| Termo (original) | Tradução | Categoria | Confiança |
|---|---|---|---:|
| pain | dor | condição médica (sintoma) | 0.96 |
| impact | impacto | condição médica (sintoma) | 0.96 |
| painful | dolorosa | condição médica (sintoma) | 0.96 |
| allergies | alergias | condição médica (sintoma) | 0.96 |
| fell | caiu | condição médica (sintoma) | 0.94 |
| scrapes | arranhões | condição médica (sintoma) | 0.91 |
| hay fever | febre do feno | condição médica (sintoma) | 0.86 |
| left side of my chest | lado esquerdo do meu peito | anatomia | 0.76 |
| fell off | caiu | condição médica (sintoma) | 0.76 |
| hurts | machuca | condição médica (sintoma) | 0.76 |

## Sintomas explicitamente negados

O paciente **negou** os itens abaixo. Registrá-los evita que sejam reinvestigados sem necessidade.

| Termo (original) | Tradução | Tipo |
|---|---|---|
| cough | tosse | DX_NAME |
| infections | infecções | DX_NAME |
| alcohol | álcool | ALCOHOL_CONSUMPTION |
| drugs | medicamentos | REC_DRUG_USE |
| marijuana | maconha | REC_DRUG_USE |

> **Atenção.** Os termos a seguir aparecem ora afirmados, ora negados em momentos diferentes da consulta: *pain*. Foram mantidos entre os achados relatados, e não entre as negações — verificar na gravação a que cada menção se refere.

## História familiar

Mencionados como ocorrências em **familiares**, não no paciente.

| Termo (original) | Tradução |
|---|---|
| diabetes | diabetes |

## Trechos que sustentam os achados

**impact** — impacto; **painful** — dolorosa; **fell** — caiu; **fell off** — caiu

> So I think it started just around 2 hours ago. I actually was riding my bicycle and just fell off on, kind of just slipped and I think I fell on that side like when I the impact was right on my chest and since then it's...

> *Acho que começou há cerca de 2 horas. Na verdade, eu estava andando de bicicleta e simplesmente caí, meio que escorregei e acho que caí daquele lado, como quando o impacto foi direto no meu peito e, desde então, é...*

**allergies** — alergias; **hay fever** — febre do feno

> Uh, not other than just a hay fever, but nothing nothing else. I have like a lot of allergies.

> *Uh, nada mais do que apenas uma febre do feno, mas nada mais. Eu tenho muitas alergias.*

**pain** — dor

> So I came to the emergency Department because I've been having pain in my chest.

> *Então eu vim para o pronto-socorro porque estava com dores no peito.*

**scrapes** — arranhões

> Besides, just I got some scrapes here and there on my hands, but nothing else.

> *Além disso, tenho alguns arranhões aqui e ali nas minhas mãos, mas nada mais.*

**left side of my chest** — lado esquerdo do meu peito

> I would say it's like the whole pretty much the left side of my chest.

> *Eu diria que é quase todo o lado esquerdo do meu peito.*

**hurts** — machuca

> If I. Think I have to take really really slow and shallow breaths. If I try to take a deep breath it just really hurts like taking those small, smaller breaths helps.

> *Se eu acho que tenho que respirar muito devagar e superficialmente. Se eu tento respirar fundo, dói muito, como se essas respirações pequenas e menores ajudassem.*

## Tom do relato

Análise de sentimento sobre a fala do paciente (Amazon Comprehend): **NEGATIVE** (negativo), com 95% de confiança na classe negativa.

Falas com maior carga negativa:

> I'd say over the last two hours it's been getting worse.

> *Eu diria que nas últimas duas horas está piorando.*

> So I think it started just around 2 hours ago. I actually was riding my bicycle and just fell off on, kind of just slipped and I think I fell on that side like when I the impact was right on my chest and since then it's...

> *Acho que começou há cerca de 2 horas. Na verdade, eu estava andando de bicicleta e simplesmente caí, meio que escorregei e acho que caí daquele lado, como quando o impacto foi direto no meu peito e, desde então, é...*

> **Como ler este indicador.** O modelo de sentimento é de propósito geral, treinado sobretudo em avaliações e redes sociais. Num relato de sintomas, o vocabulário de dor e desconforto é intrinsecamente negativo, de modo que **um resultado negativo é o esperado numa consulta e, isoladamente, diz pouco**. O indicador ganha sentido na comparação — entre casos, ou no acompanhamento do mesmo paciente ao longo do tempo. Trata-se do sentimento **do texto**, não de uma aferição do estado emocional do paciente.

## Qualidade da transcrição

A transcrição automática (Amazon Transcribe) foi comparada com a transcrição humana de referência deste dataset.

- Taxa de erro de palavra (WER): **5.4%**
- Palavras na referência: 782
- Turnos de fala identificados: 65 (referência humana: 69)

Os erros de transcrição concentram-se em convenções de escrita e palavras funcionais, não em termos clínicos.

---

Relatório gerado automaticamente a partir de: Amazon Transcribe (transcrição), Amazon Comprehend Medical (entidades clínicas), Amazon Comprehend (sentimento) e Amazon Translate (tradução). **Não substitui a avaliação de um profissional de saúde.**