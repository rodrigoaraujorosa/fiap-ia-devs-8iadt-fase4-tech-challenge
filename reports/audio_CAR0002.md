# Relatório de análise de áudio — consulta CAR0002

*Gerado em 26/07/2026 às 11:23. Documento de apoio: não substitui a avaliação clínica.*

O áudio original está em inglês. Cada termo e cada trecho aparecem no **original**, seguidos da **tradução para o português**.

## Achados relatados pelo paciente

| Termo (original) | Tradução | Categoria | Confiança |
|---|---|---|---:|
| pain | dor | condição médica (sintoma) | 0.96 |
| pressure | pressão | condição médica (sintoma) | 0.96 |
| difficulty breathing | dificuldade em respirar | condição médica (sintoma) | 0.91 |
| runny | escorrendo | condição médica (sintoma) | 0.91 |
| short of breath | falta de ar | condição médica (sintoma) | 0.88 |
| Lisinopril | Lisinopril | medicação | 0.86 |
| chest felt fine | o peito estava bem | condição médica (sintoma) | 0.84 |
| medications | medicações | exame/procedimento | 0.84 |
| Rosuvastatin | Rosuvastatina | medicação | 0.84 |
| chest | baú | anatomia | 0.80 |
| heart | coração | anatomia | 0.76 |
| feel a little bit hot | sinto um pouco de calor | condição médica (sintoma) | 0.72 |
| energy has been good | a energia tem sido boa | condição médica (sintoma) | 0.72 |
| cholesterol | colesterol | condição médica (diagnóstico) | 0.72 |
| multi vitamin | multivitamínico | medicação | 0.72 |

## Sintomas explicitamente negados

O paciente **negou** os itens abaixo. Registrá-los evita que sejam reinvestigados sem necessidade.

| Termo (original) | Tradução | Tipo |
|---|---|---|
| nausea | náusea | DX_NAME |
| vomiting | vômito | DX_NAME |
| swelling | inchando | DX_NAME |
| palpitations | palpitações | DX_NAME |
| cough | tosse | DX_NAME |
| dizziness | vertigem | DX_NAME |
| fainted | desmaiei | DX_NAME |
| chest pain | dor no peito | DX_NAME |
| rashes | erupções cutâneas | DX_NAME |
| thumping | batendo | DX_NAME |
| allergies | alergias | ALLERGIES |
| alcohol | álcool | ALCOHOL_CONSUMPTION |
| marijuana | maconha | REC_DRUG_USE |

## Trechos que sustentam os achados

**Lisinopril** — Lisinopril; **medications** — medicações; **Rosuvastatin** — Rosuvastatina; **multi vitamin** — multivitamínico

> Um, I do take medications for both blood pressure and cholesterol, Rosuvastatin and um Lisinopril and I take a multi vitamin.

> *Eu tomo medicamentos para pressão arterial e colesterol, rosuvastatina e lisinopril e tomo um multivitamínico.*

**pain** — dor; **chest** — baú

> Yeah, I have this pain in my chest.

> *Sim, eu tenho essa dor no meu peito.*

**difficulty breathing** — dificuldade em respirar; **short of breath** — falta de ar

> Uh, I've felt a little bit uh short of breath or having difficulty breathing since yesterday when the sorry since the pain started, but uh just the difficulty breathing.

> *Uh, eu senti um pouco de falta de ar ou dificuldade em respirar desde ontem, quando a dor começou, mas... só a dificuldade em respirar.*

**heart** — coração; **energy has been good** — a energia tem sido boa

> No, my energy has been good. D; Have you been having any kind of thumping or palpitations or feel like your heart has been racing at all?

> *Não, minha energia tem sido boa. D; Você está tendo algum tipo de batimento ou palpitações ou sente que seu coração está acelerado?*

**pressure** — pressão

> It feels dull. I feel like there's a lot of pressure on my chest.

> *Parece monótono. Sinto que há muita pressão no meu peito.*

**runny** — escorrendo

> Uh few weeks ago I was a little runny, but that went away on its own. I haven't had any cough.

> *Algumas semanas atrás eu estava um pouco irritado, mas isso desapareceu sozinho. Eu não tive nenhuma tosse.*

## Tom do relato

Análise de sentimento sobre a fala do paciente (Amazon Comprehend): **NEGATIVE** (negativo), com 98% de confiança na classe negativa.

Falas com maior carga negativa:

> It feels dull. I feel like there's a lot of pressure on my chest.

> *Parece monótono. Sinto que há muita pressão no meu peito.*

> Uh. I think it's a bit bit worse if I'm moving around or when I was walking in here. I think it it made it a bit worse, but nothing has seemed to make it any better since it starting.

> *Ah. Acho que é um pouco pior se eu estiver me movendo ou entrando aqui. Acho que piorou um pouco as coisas, mas nada pareceu melhorá-las desde o início.*

> **Como ler este indicador.** O modelo de sentimento é de propósito geral, treinado sobretudo em avaliações e redes sociais. Num relato de sintomas, o vocabulário de dor e desconforto é intrinsecamente negativo, de modo que **um resultado negativo é o esperado numa consulta e, isoladamente, diz pouco**. O indicador ganha sentido na comparação — entre casos, ou no acompanhamento do mesmo paciente ao longo do tempo. Trata-se do sentimento **do texto**, não de uma aferição do estado emocional do paciente.

## Qualidade da transcrição

A transcrição automática (Amazon Transcribe) foi comparada com a transcrição humana de referência deste dataset.

- Taxa de erro de palavra (WER): **5.0%**
- Palavras na referência: 1043
- Turnos de fala identificados: 81 (referência humana: 82)

Os erros de transcrição concentram-se em convenções de escrita e palavras funcionais, não em termos clínicos.

---

Relatório gerado automaticamente a partir de: Amazon Transcribe (transcrição), Amazon Comprehend Medical (entidades clínicas), Amazon Comprehend (sentimento) e Amazon Translate (tradução). **Não substitui a avaliação de um profissional de saúde.**