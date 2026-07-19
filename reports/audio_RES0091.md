# Relatório de análise de áudio — consulta RES0091

*Gerado em 18/07/2026 às 21:56. Documento de apoio: não substitui a avaliação clínica.*

O áudio original está em inglês. Cada termo e cada trecho aparecem no **original**, seguidos da **tradução para o português**.

## Achados relatados pelo paciente

| Termo (original) | Tradução | Categoria | Confiança |
|---|---|---|---:|
| headache | enxaqueca | condição médica (sintoma) | 0.96 |
| fluid | fluido | condição médica (sintoma) | 0.96 |
| healthy | saudável | condição médica (sintoma) | 0.96 |
| runny nose | corrimento nasal | condição médica (sintoma) | 0.91 |
| stuffy nose | nariz entupido | condição médica (sintoma) | 0.88 |
| arm | braço | anatomia | 0.86 |
| nose | nariz | anatomia | 0.82 |
| arm fracture | fratura do braço | condição médica (diagnóstico) | 0.82 |
| wine | vinho | comportamental/social | 0.80 |

## Sintomas explicitamente negados

O paciente **negou** os itens abaixo. Registrá-los evita que sejam reinvestigados sem necessidade.

| Termo (original) | Tradução | Tipo |
|---|---|---|
| coughing | tossindo | DX_NAME |
| dizziness | vertigem | DX_NAME |
| allergies | alergias | ALLERGIES |

## Trechos que sustentam os achados

**runny nose** — corrimento nasal; **stuffy nose** — nariz entupido; **nose** — nariz

> Uhm, so I've had this runny nose, well first it was a stuffy nose, but now I've had this runny nose for the past week and a half and it doesn't seem to be getting any better. So I was just wondering if you could give me...

> *Uhm, então eu tive esse corrimento nasal, bem, primeiro foi um nariz entupido, mas agora eu tive esse corrimento nasal na última semana e meia e não parece estar melhorando. Então eu só queria saber se você poderia me dar...*

**headache** — enxaqueca; **fluid** — fluido

> No dizziness, I feel like sometimes I have a headache though, because I feel like there's a lot of fluid or something backed up into my sinuses.

> *Sem tontura, mas às vezes sinto que tenho dor de cabeça, porque sinto que há muito líquido ou algo acumulado nos meus seios nasais.*

**arm** — braço; **arm fracture** — fratura do braço

> I had an arm fracture when I was younger, but that's it.

> *Eu tive uma fratura no braço quando era mais jovem, mas só isso.*

**healthy** — saudável

> Uhm, no, I'm healthy otherwise.

> *Uhm, não, eu sou saudável de outra forma.*

**wine** — vinho

> Once in a while I'll have like a glass of wine every week.

> *De vez em quando, tomo uma taça de vinho toda semana.*

## Tom do relato

Análise de sentimento sobre a fala do paciente (Amazon Comprehend): **NEGATIVE** (negativo), com 88% de confiança na classe negativa.

Falas com maior carga negativa:

> Uhm, so I've had this runny nose, well first it was a stuffy nose, but now I've had this runny nose for the past week and a half and it doesn't seem to be getting any better. So I was just wondering if you could give me...

> *Uhm, então eu tive esse corrimento nasal, bem, primeiro foi um nariz entupido, mas agora eu tive esse corrimento nasal na última semana e meia e não parece estar melhorando. Então eu só queria saber se você poderia me dar...*

> Uh, no, what can I do about this runny nose?

> *Não, o que posso fazer com esse corrimento nasal?*

> **Como ler este indicador.** O modelo de sentimento é de propósito geral, treinado sobretudo em avaliações e redes sociais. Num relato de sintomas, o vocabulário de dor e desconforto é intrinsecamente negativo, de modo que **um resultado negativo é o esperado numa consulta e, isoladamente, diz pouco**. O indicador ganha sentido na comparação — entre casos, ou no acompanhamento do mesmo paciente ao longo do tempo. Trata-se do sentimento **do texto**, não de uma aferição do estado emocional do paciente.

## Qualidade da transcrição

A transcrição automática (Amazon Transcribe) foi comparada com a transcrição humana de referência deste dataset.

- Taxa de erro de palavra (WER): **4.1%**
- Palavras na referência: 873
- Turnos de fala identificados: 79 (referência humana: 79)

Os erros de transcrição concentram-se em convenções de escrita e palavras funcionais, não em termos clínicos.

---

Relatório gerado automaticamente a partir de: Amazon Transcribe (transcrição), Amazon Comprehend Medical (entidades clínicas), Amazon Comprehend (sentimento) e Amazon Translate (tradução). **Não substitui a avaliação de um profissional de saúde.**