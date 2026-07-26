# Relatório de análise de áudio — consulta CAR0001

*Gerado em 26/07/2026 às 11:05. Documento de apoio: não substitui a avaliação clínica.*

O áudio original está em inglês. Cada termo e cada trecho aparecem no **original**, seguidos da **tradução para o português**.

## Achados relatados pelo paciente

| Termo (original) | Tradução | Categoria | Confiança |
|---|---|---|---:|
| lightheaded | vertigens | condição médica (sintoma) | 0.92 |
| chest pain | dor no peito | condição médica (sintoma) | 0.91 |
| trouble breathing | dificuldade em respirar | condição médica (sintoma) | 0.89 |
| swollen | inchada | condição médica (sintoma) | 0.88 |
| diet | dieta | exame/procedimento | 0.84 |
| neck | pescoço | anatomia | 0.82 |
| left side of my chest | lado esquerdo do meu peito | anatomia | 0.80 |
| issues breathing | problemas respiratórios | condição médica (sintoma) | 0.80 |
| heart | coração | anatomia | 0.80 |
| male | macho | comportamental/social | 0.76 |
| chest | baú | anatomia | 0.72 |
| activity | atividade | exame/procedimento | 0.72 |

## Sintomas explicitamente negados

O paciente **negou** os itens abaixo. Registrá-los evita que sejam reinvestigados sem necessidade.

| Termo (original) | Tradução | Tipo |
|---|---|---|
| rashes | erupções cutâneas | DX_NAME |
| strokes | derrames | DX_NAME |

## História familiar

Mencionados como ocorrências em **familiares**, não no paciente.

| Termo (original) | Tradução |
|---|---|
| heart attack | ataque cardíaco |
| strokes | derrames |

## Trechos que sustentam os achados

**lightheaded** — vertigens; **trouble breathing** — dificuldade em respirar

> I feel a little lightheaded and I'm having some trouble breathing.

> *Sinto um pouco de tontura e estou tendo problemas para respirar.*

**chest pain** — dor no peito; **chest** — baú

> Sure, I'm I'm just having a lot of chest pain and and so I thought I should get it checked out.

> *Claro, só estou com muita dor no peito, então achei que deveria dar uma olhada.*

**swollen** — inchada; **neck** — pescoço

> No rashes, but I guess like my neck seems to be a little swollen.

> *Sem erupções cutâneas, mas acho que meu pescoço parece estar um pouco inchado.*

**diet** — dieta; **activity** — atividade

> Sure, I try to eat healthy for dinner at least, but most of my lunches are, uh I eat out. And then in terms of exercise, I try to exercise every other day, I run for about half an hour. D; OK, well that's great that...

> *Claro, eu tento comer comida saudável no jantar, pelo menos, mas a maioria dos meus almoços é, uh, eu como fora. E então, em termos de exercício, tento me exercitar todos os dias, corro por cerca de meia hora. D; OK, bem, isso é ótimo que...*

**left side of my chest** — lado esquerdo do meu peito

> It's located on the left side of my chest.

> *Está localizado no lado esquerdo do meu peito.*

**issues breathing** — problemas respiratórios

> Just from the from having issues breathing.

> *Só por ter problemas respiratórios.*

## Tom do relato

Análise de sentimento sobre a fala do paciente (Amazon Comprehend): **POSITIVE** (positivo), com 0% de confiança na classe negativa.

Falas com maior carga negativa:

> I'd say it's like a seven or eight. It's pretty bad.

> *Eu diria que é como um sete ou oito. É muito ruim.*

> I feel a little lightheaded and I'm having some trouble breathing.

> *Sinto um pouco de tontura e estou tendo problemas para respirar.*

> **Como ler este indicador.** O modelo de sentimento é de propósito geral, treinado sobretudo em avaliações e redes sociais. Num relato de sintomas, o vocabulário de dor e desconforto é intrinsecamente negativo, de modo que **um resultado negativo é o esperado numa consulta e, isoladamente, diz pouco**. O indicador ganha sentido na comparação — entre casos, ou no acompanhamento do mesmo paciente ao longo do tempo. Trata-se do sentimento **do texto**, não de uma aferição do estado emocional do paciente.

## Qualidade da transcrição

A transcrição automática (Amazon Transcribe) foi comparada com a transcrição humana de referência deste dataset.

- Taxa de erro de palavra (WER): **6.6%**
- Palavras na referência: 1028
- Turnos de fala identificados: 66 (referência humana: 121)

Os erros de transcrição concentram-se em convenções de escrita e palavras funcionais, não em termos clínicos.

---

Relatório gerado automaticamente a partir de: Amazon Transcribe (transcrição), Amazon Comprehend Medical (entidades clínicas), Amazon Comprehend (sentimento) e Amazon Translate (tradução). **Não substitui a avaliação de um profissional de saúde.**