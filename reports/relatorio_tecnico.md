# 📄 Relatório Técnico — Tech Challenge Fase 4

> Sistema de Monitoramento Hospitalar Multimodal · PosTech FIAP · 8IADT

## 1. Descrição do fluxo multimodal
_(Como vídeo, áudio e séries temporais são processados e como os alertas se integram.)_

## 2. Modelos aplicados por tipo de dado

### 2.1 🎥 Vídeo
- Modelo: OpenPose (BODY_25) — estimação de pose 2D
- Dataset: REHAB24-6 (reabilitação física, RGB + rótulos correto/incorreto)
- Abordagem: _..._

### 2.2 🎙️ Áudio
- Azure Speech-to-Text: _..._
- Azure Text Analytics: _..._
- Biomarcadores acústicos: _..._
- Dataset: Coswara

### 2.3 📈 Detecção de Anomalias
- Sinais vitais (PhysioNet Challenge 2019): IsolationForest — _..._
- Movimentação (UCI HAR): IsolationForest — _..._
- Prescrições (Synthea): _..._

## 3. Resultados obtidos
_(Métricas, tabelas.)_

## 4. Exemplos de anomalias detectadas
_(Figuras em `reports/figures/`, casos comentados.)_

## 5. Integração Azure e fluxo de alerta
_(Como o alerta chega à equipe médica.)_

## 6. Conclusão
_..._
