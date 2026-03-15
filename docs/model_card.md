# Model Card v2 — Predição de Readmissão Hospitalar em 30 Dias

> Versão: 1.0.0 | Última atualização: 2026-03-15 | Idioma: Português (BR)

---

## 1. Visão Geral do Modelo

| Campo | Descrição |
|---|---|
| **Nome** | Hospital Readmission Ensemble v1.0 |
| **Tipo** | Classificação binária — Ensemble (DNN + XGBoost) |
| **Tarefa** | Prever risco de readmissão hospitalar em 30 dias |
| **Versão** | 1.0.0 |
| **Licença** | MIT |
| **Contato** | [GitHub Issues](https://github.com/user/hospital-readmission-ml/issues) |

---

## 2. Uso Pretendido

### 2.1 Uso Primário
Auxiliar equipes clínicas na identificação de pacientes com alto risco de readmissão hospitalar antes da alta, permitindo intervenções preventivas como:
- Contato ativo em 7 dias pós-alta (alto risco)
- Consulta ambulatorial em 14 dias (risco moderado)
- Alta padrão com orientações (baixo risco)

### 2.2 Usuários Pretendidos
- Médicos e enfermeiros responsáveis pela alta hospitalar
- Gestores de programas de redução de readmissão
- Pesquisadores em saúde pública e epidemiologia

### 2.3 Uso Impróprio — O que este modelo NÃO deve fazer
- ❌ Substituir o julgamento clínico do profissional de saúde
- ❌ Ser usado como único critério para decisões de internação ou alta
- ❌ Aplicar-se a populações fora do perfil de treinamento (ver Seção 5)
- ❌ Ser implantado sem validação prospectiva em ambiente clínico real
- ❌ Processar dados de pacientes sem consentimento conforme LGPD

---

## 3. Fatores Relevantes

### 3.1 Fatores Demográficos Cobertos
- Idade: 0–100 anos (melhor performance em 50–80 anos)
- Gênero: Masculino, Feminino
- Diagnóstico primário: 8 categorias (Circulatório, Respiratório, Digestivo, Diabetes, Lesão, Musculoesquelético, Geniturinário, Outro)

### 3.2 Fatores Clínicos Considerados
- Tempo de internação (1–14 dias)
- Número de medicamentos, procedimentos, diagnósticos
- Exames de HbA1c e glicose sérica
- Uso de insulina e mudanças medicamentosas
- Histórico de visitas ambulatoriais, emergenciais e internações

### 3.3 Fora do Escopo
- Pacientes pediátricos (< 18 anos) — performance não validada
- Internações em UTI
- Pacientes oncológicos (diagnóstico não coberto)
- Gravidez e complicações obstétricas
- Doenças raras ou condições não classificáveis nas 8 categorias

---

## 4. Métricas de Avaliação

### 4.1 Métricas Globais (conjunto de teste — 2.000 amostras)

| Métrica | Threshold 0.50 | **Threshold 0.37** (otimizado) |
|---|---|---|
| ROC-AUC | 0.6882 | 0.6882 |
| Average Precision | 0.7603 | 0.7603 |
| F1-Score | 0.6887 | **0.7609** |
| Precisão | 0.7104 | 0.6512 |
| Recall (Sensibilidade) | 0.6683 | **0.9150** |
| Especificidade | ~0.265 | ~0.265 |
| Brier Score | ~0.22 | ~0.22 |

> **Justificativa do threshold 0.37:** Em contexto clínico, falsos negativos (pacientes de alto risco não identificados) têm maior custo que falsos positivos. O threshold 0.37 maximiza o recall (91,5%), priorizando não perder pacientes em risco.

### 4.2 Comparação de Modelos

| Modelo | ROC-AUC | F1 (t=0.37) | Recall |
|---|---|---|---|
| DNN (Keras) | 0.6848 | — | — |
| XGBoost | 0.6817 | — | — |
| **Ensemble (55% DNN + 45% XGB)** | **0.6882** | **0.7609** | **0.915** |

### 4.3 Calibração
- Brier Score bruto (antes da calibração): ~0.22
- Após regressão isotônica: a ser calculado após `python model/calibrate.py`
- Curva de confiabilidade: `model/calibration_curve.png`

### 4.4 Análise por Subgrupos (Fairness)
Detalhes completos em `model/fairness_report.json`. Limites de alerta:
- AUC mínimo por subgrupo: 0.60
- Delta máximo de TPR entre subgrupos: 0.20

---

## 5. Dados de Treinamento

### 5.1 Fonte Primária (padrão)
**Dataset sintético** gerado em `data/generate_data.py`:
- 10.001 amostras
- Baseado nas distribuições do UCI Diabetes 130-US Hospitals Dataset
- Taxa de readmissão: ~55% (classe desbalanceada)
- Gerado com `random_state=42` para reprodutibilidade

### 5.2 Fonte Alternativa (optional)
**UCI Diabetes 130-US Hospitals Dataset** (1999–2008):
- ~101.766 encontros hospitalares
- Hospitais nos EUA
- Processado via `data/process_real_data.py`
- **Não está incluído no repositório** — requer download manual

### 5.3 Limitações dos Dados
- Dados históricos (1999–2008): práticas clínicas e medicamentos mudaram
- Cobertura majoritariamente de pacientes diabéticos nos EUA
- Dados sintéticos podem não capturar todas as correlações clínicas reais
- Ausência de variáveis socioeconômicas (moradia, suporte familiar, etc.)

---

## 6. Dados de Avaliação

- Split estratificado: 80% treino / 15% validação / 5% teste (durante treinamento)
- Para calibração: 65% / 15% calibração / 20% teste (em `model/calibrate.py`)
- Não houve validação em dataset externo até esta versão

---

## 7. Análise Quantitativa

### 7.1 Importância de Features (XGBoost — top 5)
1. `number_inpatient` — número de internações anteriores
2. `num_medications` — número de medicamentos na alta
3. `time_in_hospital` — duração da internação atual
4. `num_diagnoses` — número de diagnósticos
5. `hba1c_result` — resultado do exame HbA1c

### 7.2 Features Derivadas
- `risk_score` = 2×internações + emergências + indicadores HbA1c/glicose
- `medication_complexity` = medicamentos × diagnósticos
- `hospital_utilization` = ambulatorial + emergência + internação

---

## 8. Considerações Éticas

### 8.1 Viés e Equidade
- O modelo foi treinado majoritariamente com dados de pacientes diabéticos — pode não generalizar igualmente para outras populações
- Variáveis socioeconômicas (renda, raça, acesso a saúde) não estão incluídas, podendo mascarar disparidades existentes
- É obrigatória a análise de fairness (`python model/fairness.py`) antes de qualquer implantação clínica

### 8.2 Privacidade e LGPD
- A API **não armazena dados pessoais identificáveis** nos logs de auditoria
- Cada predição é registrada com hash SHA-256 das features (não reversível para PII)
- Base legal: Art. 11, II, f da LGPD (tutela da saúde)
- Detalhes completos em `docs/lgpd_conformidade.md`

### 8.3 Responsabilidade Clínica
- Conforme **CFM Resolução 2.227/2018**, sistemas de IA em medicina devem auxiliar, nunca substituir, o médico responsável
- O profissional de saúde retém integralmente a responsabilidade pela decisão clínica
- A predição deve ser apresentada junto com os fatores que contribuíram para ela (via SHAP)

---

## 9. Advertências e Recomendações

### 9.1 Antes de Usar em Produção
- [ ] Executar validação prospectiva com protocolo aprovado por CEP (ver `docs/guia_validacao_clinica.md`)
- [ ] Executar `python model/calibrate.py` e ativar `USE_CALIBRATED_MODEL=true`
- [ ] Executar `python model/fairness.py` e verificar ausência de alertas
- [ ] Executar `python model/compute_baseline.py` para gerar checksums e baseline
- [ ] Configurar `SECRET_KEY` forte (mínimo 32 chars aleatórios)
- [ ] Configurar HTTPS (TLS) antes de processar dados reais
- [ ] Criar usuários com roles adequadas (`clinician` para uso clínico, `admin` para gestão)
- [ ] Revisar `docs/lgpd_conformidade.md` com o encarregado de dados (DPO)

### 9.2 Monitoramento Contínuo
- Verificar métricas Prometheus regularmente (GET /metrics)
- Monitorar PSI de deriva de dados (alert PSI > 0.2)
- Re-treinar se ROC-AUC cair abaixo de 0.65 em validação contínua
- Auditar logs regularmente (GET /audit/summary — admin)

---

## 10. Informações Regulatórias

| Regulamento | Aplicabilidade | Status |
|---|---|---|
| **LGPD** (Lei 13.709/2018) | Obrigatório — dados de saúde são dados sensíveis | Implementado (auditoria, base legal, retenção) |
| **CFM Res. 2.227/2018** | Recomendado — IA como auxílio ao médico | Implementado (disclaimers, SHAP explicabilidade) |
| **ANVISA RDC 657/2022** | Aplicável se classificado como SaMD (dispositivo médico) | Requer avaliação jurídica e regulatória |
| **Resolução CNS 466/2012** | Obrigatório para pesquisa com dados humanos | Ver guia de validação clínica |

> ⚠ **Aviso Legal**: Este software é fornecido para fins educacionais e de pesquisa. Para uso clínico real no Brasil, é necessário assessoria jurídica especializada em direito sanitário e eventual registro na ANVISA como Software como Dispositivo Médico (SaMD).
