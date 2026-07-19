# Relatório de análise de áudio — consulta MSK0018

*Gerado em 18/07/2026 às 22:24. Documento de apoio: não substitui a avaliação clínica.*

O áudio original está em inglês. Cada termo e cada trecho aparecem no **original**, seguidos da **tradução para o português**.

## Achados relatados pelo paciente

| Termo (original) | Tradução | Categoria | Confiança |
|---|---|---|---:|
| fell | caiu | condição médica (sintoma) | 0.96 |
| pain | dor | condição médica (sintoma) | 0.96 |
| weakness | fraqueza | condição médica (sintoma) | 0.96 |
| healthy | saudável | condição médica (sintoma) | 0.96 |
| elbow | cotovelo | anatomia | 0.91 |
| hand | mão | anatomia | 0.89 |
| shoulder's dropped | o ombro caiu | condição médica (sintoma) | 0.88 |
| right shoulder | ombro direito | anatomia | 0.84 |
| shoulder | ombro | anatomia | 0.76 |
| arm | braço | anatomia | 0.72 |
| divots | divots | condição médica (sintoma) | 0.72 |
| hurts | machuca | condição médica (sintoma) | 0.72 |

## Sintomas explicitamente negados

O paciente **negou** os itens abaixo. Registrá-los evita que sejam reinvestigados sem necessidade.

| Termo (original) | Tradução | Tipo |
|---|---|---|
| swelling | inchando | DX_NAME |
| tingling | formigamento | DX_NAME |
| redness | vermelhidão | DX_NAME |
| deformity | deformidade | DX_NAME |

> **Atenção.** Os termos a seguir aparecem ora afirmados, ora negados em momentos diferentes da consulta: *pain*. Foram mantidos entre os achados relatados, e não entre as negações — verificar na gravação a que cada menção se refere.

## Trechos que sustentam os achados

**fell** — caiu; **pain** — dor; **right shoulder** — ombro direito; **shoulder** — ombro

> I just came in because of I just got into like this uh, incident while I was playing rugby. I kind of got tackled and then fell onto my right shoulder. Um, and after I got up, I haven't, I've just been, this just...

> *Acabei de entrar porque acabei de entrar nesse incidente enquanto jogava rúgbi. Eu meio que fui atacado e depois caí no meu ombro direito. Hum, e depois que me levantei, eu não, eu só estava, isso só...*

**elbow** — cotovelo; **hand** — mão

> I could move my fingers and hand and elbow, but um, elbow a little bit, but like uh, yeah, I can't move my shoulder whatsoever.

> *Eu podia mover meus dedos, mão e cotovelo, mas um pouco de cotovelo, mas tipo, sim, eu não consigo mover meu ombro de jeito nenhum.*

**shoulder's dropped** — o ombro caiu; **divots** — divots

> Um, deformity, I, I do feel like kind of uh, like my shoulder's dropped and like there's just kind of this uh, little, like divots, where like the shoulder should be.

> *Hum, deformidade, eu, eu me sinto como se meu ombro estivesse caído e como se houvesse uma espécie de, uh, pequenas, como se o ombro estivesse.*

**weakness** — fraqueza

> Um, in that area, I don't know if it's weakness or pain, but I just can't move it up and I, I, I do feel some like numbness just above my shoulder. No tingling though.

> *Hum, nessa área, eu não sei se é fraqueza ou dor, mas eu simplesmente não consigo movê-la para cima e eu, eu, eu sinto um pouco de dormência logo acima do meu ombro. Mas sem formigamento.*

**healthy** — saudável

> No. Yeah, otherwise I'm healthy. I've never been to the doctor much.

> *Não. Sim, caso contrário, estou saudável. Nunca fui muito ao médico.*

**arm** — braço

> Uh, mostly towards my shoulder, um, kind of down towards my upper arm as well.

> *Uh, principalmente em direção ao meu ombro, um pouco abaixo do meu braço também.*

## Tom do relato

Análise de sentimento sobre a fala do paciente (Amazon Comprehend): **NEGATIVE** (negativo), com 97% de confiança na classe negativa.

Falas com maior carga negativa:

> No, I, I can't. I can't move it at all.

> *Não, eu, eu não posso. Não consigo movê-lo de jeito nenhum.*

> Um, um not anything that I've tried that's made it better. Worse is probably just if I try to move it at all.

> *Hum, hum, nada que eu tenha tentado que o tornou melhor. O pior é provavelmente se eu tentar movê-lo.*

> **Como ler este indicador.** O modelo de sentimento é de propósito geral, treinado sobretudo em avaliações e redes sociais. Num relato de sintomas, o vocabulário de dor e desconforto é intrinsecamente negativo, de modo que **um resultado negativo é o esperado numa consulta e, isoladamente, diz pouco**. O indicador ganha sentido na comparação — entre casos, ou no acompanhamento do mesmo paciente ao longo do tempo. Trata-se do sentimento **do texto**, não de uma aferição do estado emocional do paciente.

## Qualidade da transcrição

A transcrição automática (Amazon Transcribe) foi comparada com a transcrição humana de referência deste dataset.

- Taxa de erro de palavra (WER): **7.5%**
- Palavras na referência: 943
- Turnos de fala identificados: 73 (referência humana: 86)

Os erros de transcrição concentram-se em convenções de escrita e palavras funcionais, não em termos clínicos.

---

Relatório gerado automaticamente a partir de: Amazon Transcribe (transcrição), Amazon Comprehend Medical (entidades clínicas), Amazon Comprehend (sentimento) e Amazon Translate (tradução). **Não substitui a avaliação de um profissional de saúde.**