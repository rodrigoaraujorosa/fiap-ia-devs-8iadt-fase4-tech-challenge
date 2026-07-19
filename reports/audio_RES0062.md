# Relatório de análise de áudio — consulta RES0062

*Gerado em 18/07/2026 às 22:57. Documento de apoio: não substitui a avaliação clínica.*

O áudio original está em inglês. Cada termo e cada trecho aparecem no **original**, seguidos da **tradução para o português**.

## Achados relatados pelo paciente

| Termo (original) | Tradução | Categoria | Confiança |
|---|---|---|---:|
| cough | tosse | condição médica (sintoma) | 0.96 |
| tired | cansada | condição médica (sintoma) | 0.96 |
| drink | beber | condição médica (sintoma) | 0.96 |
| shortness of breath | falta de ar | condição médica (sintoma) | 0.94 |
| wheezing | chiado | condição médica (sintoma) | 0.94 |
| achy | dolorido | condição médica (sintoma) | 0.89 |
| hacking | hackeando | condição médica (sintoma) | 0.88 |
| short of breath | falta de ar | condição médica (sintoma) | 0.86 |
| atrial | atriais | anatomia | 0.86 |
| rash | erupção cutânea | condição médica (sintoma) | 0.86 |
| throat | garganta | anatomia | 0.84 |
| throat felt OK. | a garganta estava bem. | condição médica (sintoma) | 0.82 |
| Metoprolol | Metoprolol | medicação | 0.82 |
| heart was racing | o coração estava acelerado | condição médica (sintoma) | 0.82 |
| lung | pulmão | anatomia | 0.82 |
| belly | barriga | anatomia | 0.80 |
| atrial fibrillation | fibrilação atrial | condição médica (diagnóstico) | 0.80 |
| haven't been able to smell | não consegui cheirar | condição médica (sintoma) | 0.76 |
| beta blocker | bloqueador beta | exame/procedimento | 0.76 |
| heart rate | frequência cardíaca | exame/procedimento | 0.76 |
| inguinal hernia repair | reparo de hérnia inguinal | exame/procedimento | 0.76 |
| statin | estatina | exame/procedimento | 0.76 |
| cigarettes | cigarros | comportamental/social | 0.76 |
| feeling a little hot | sentindo um pouco de calor | condição médica (sintoma) | 0.76 |
| nose | nariz | anatomia | 0.72 |

## Sintomas explicitamente negados

O paciente **negou** os itens abaixo. Registrá-los evita que sejam reinvestigados sem necessidade.

| Termo (original) | Tradução | Tipo |
|---|---|---|
| blood | sangue | DX_NAME |
| dizziness | vertigem | DX_NAME |
| weak | fraco | DX_NAME |
| diarrhea | diarreia | DX_NAME |
| constipation | Prisão de ventre | DX_NAME |
| pain | dor | DX_NAME |
| symptoms | sintomas | DX_NAME |
| loss of taste | perda do paladar | DX_NAME |
| fainted | desmaiei | DX_NAME |
| blow my nose | assoar meu nariz | DX_NAME |
| felt good | me senti bem | DX_NAME |
| antibiotics | antibióticos | TREATMENT_NAME |

## História familiar

Mencionados como ocorrências em **familiares**, não no paciente.

| Termo (original) | Tradução |
|---|---|
| lung cancer | câncer de pulmão |
| cancers | cânceres |

## Trechos que sustentam os achados

**atrial** — atriais; **Metoprolol** — Metoprolol; **atrial fibrillation** — fibrilação atrial

> Yeah, I have atrial fibrillation. And I do take Metoprolol for that.

> *Sim, eu tenho fibrilação atrial. E eu tomo metoprolol para isso.*

**cough** — tosse; **short of breath** — falta de ar

> Well, I've had this cough but the biggest issue is that I've been really short of breath.

> *Bem, eu tive essa tosse, mas o maior problema é que estou com muita falta de ar.*

**achy** — dolorido; **belly** — barriga

> Oh yeah, I don't know why, yeah, my belly's been a bit achy maybe.

> *Ah, sim, eu não sei por que, sim, talvez minha barriga esteja um pouco dolorida.*

**throat** — garganta; **throat felt OK.** — a garganta estava bem.

> No, my throat felt OK.

> *Não, minha garganta estava bem.*

**heart was racing** — o coração estava acelerado; **beta blocker** — bloqueador beta

> It was like 4 years ago I think it was. It was having that like, felt like my heart was racing you were asking about earlier and they did an ECG and I was told I had atrial fibrillation. So yeah, I've been on a beta...

> *Acho que foi há 4 anos. Foi como se meu coração estivesse acelerado sobre o qual você estava perguntando mais cedo e eles fizeram um ECG e me disseram que eu tinha fibrilação atrial. Então, sim, eu estive em uma versão beta...*

**tired** — cansada

> As soon as I breath, well, I just haven't felt good like, tired, kind of weakish.

> *Assim que respiro, bem, eu simplesmente não me sinto bem, cansada, meio fraca.*

## Tom do relato

Análise de sentimento sobre a fala do paciente (Amazon Comprehend): **NEGATIVE** (negativo), com 75% de confiança na classe negativa.

Falas com maior carga negativa:

> As soon as I breath, well, I just haven't felt good like, tired, kind of weakish.

> *Assim que respiro, bem, eu simplesmente não me sinto bem, cansada, meio fraca.*

> No the cough has been kind of going on all day. Um, right when I get up in the morning and goes on all night, it's been getting worse to these last few days.

> *Não, a tosse está meio que acontecendo o dia todo. Bem, quando eu me levanto de manhã e fico a noite toda, está piorando nos últimos dias.*

> **Como ler este indicador.** O modelo de sentimento é de propósito geral, treinado sobretudo em avaliações e redes sociais. Num relato de sintomas, o vocabulário de dor e desconforto é intrinsecamente negativo, de modo que **um resultado negativo é o esperado numa consulta e, isoladamente, diz pouco**. O indicador ganha sentido na comparação — entre casos, ou no acompanhamento do mesmo paciente ao longo do tempo. Trata-se do sentimento **do texto**, não de uma aferição do estado emocional do paciente.

## Qualidade da transcrição

A transcrição automática (Amazon Transcribe) foi comparada com a transcrição humana de referência deste dataset.

- Taxa de erro de palavra (WER): **10.8%**
- Palavras na referência: 1910
- Turnos de fala identificados: 128 (referência humana: 130)

Os erros de transcrição concentram-se em convenções de escrita e palavras funcionais, não em termos clínicos.

---

Relatório gerado automaticamente a partir de: Amazon Transcribe (transcrição), Amazon Comprehend Medical (entidades clínicas), Amazon Comprehend (sentimento) e Amazon Translate (tradução). **Não substitui a avaliação de um profissional de saúde.**