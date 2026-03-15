# Guia de Validação Clínica

## Hospital Readmission Prediction Model — Protocolo de Validação Prospectiva

**Versão:** 1.0 | **Data:** 2026-03-15

---

## 1. Por Que Validação Clínica é Necessária

O modelo foi treinado com dados sintéticos e/ou históricos (UCI dataset 1999–2008). Antes de uso clínico real, é necessário demonstrar que:

1. A performance reportada (AUC 0.69, Recall 91.5%) se mantém na população-alvo do hospital
2. Não há viés sistemático para subgrupos de pacientes atendidos localmente
3. O impacto clínico (redução de readmissões) justifica o custo de implementação
4. O sistema está em conformidade com regulamentações locais

---

## 2. Classificação Regulatória

### 2.1 ANVISA RDC 657/2022 — Software como Dispositivo Médico (SaMD)
Este sistema pode ser classificado como SaMD de **Classe II** (risco moderado), pois:
- Fornece informação clínica para apoio à decisão
- Não é o único meio de diagnóstico ou terapia
- Erros podem causar dano não imediato ao paciente

**Ação requerida:** Consultar assessoria regulatória para determinar se é necessário registro na ANVISA antes da implantação clínica.

### 2.2 Comitê de Ética em Pesquisa (CEP)
Qualquer estudo de validação com dados de pacientes reais requer aprovação do CEP conforme:
- **Resolução CNS 466/2012** — pesquisa com seres humanos
- **Resolução CNS 580/2018** — pesquisa em banco de dados
- **LGPD Art. 11, II, e** — pesquisa científica com dados de saúde

---

## 3. Protocolo de Validação Proposto

### 3.1 Desenho do Estudo
- **Tipo:** Estudo de coorte prospectivo, não intervencionista (fase 1) → ensaio clínico randomizado por cluster (fase 2)
- **Duração mínima:** 6 meses (fase 1) + 12 meses (fase 2)
- **Desfecho primário:** Readmissão em 30 dias (qualquer causa)
- **Desfecho secundário:** Readmissão evitável, tempo até readmissão, custo por QALY

### 3.2 Cálculo de Tamanho Amostral
Para detectar AUC ≥ 0.70 com:
- Poder estatístico: 80%
- Nível de significância: α = 0.05
- Prevalência de readmissão esperada: ~15–20%
- **N mínimo estimado: 500–800 pacientes** (calcular com dados locais)

Fórmula recomendada: método de Hanley-McNeil para comparação de AUC (1982).

```
# Cálculo de n usando statsmodels (Python):
from statsmodels.stats.power import TTestIndPower
# Adaptar para AUC usando: n ≈ (z_alpha + z_beta)^2 / (AUC - 0.5)^2 * constante
```

### 3.3 Critérios de Inclusão
- Pacientes adultos (≥ 18 anos) internados por qualquer causa
- Internação completa (alta hospitalar, não óbito ou transferência)
- Diagnóstico primário classificável nas 8 categorias do modelo
- Dados suficientes para preencher todos os 16 campos de entrada

### 3.4 Critérios de Exclusão
- Pacientes em cuidados paliativos
- Pacientes oncológicos com quimioterapia ativa
- Internações programadas (cirurgias eletivas sem comorbidades)
- Pacientes sem dados de exames laboratoriais (HbA1c, glicose)

---

## 4. Fases da Validação

### Fase 1 — Validação de Performance (0–6 meses)
**Objetivo:** Confirmar que AUC ≥ 0.65 na população local

**Procedimento:**
1. Coleta prospectiva de dados nos campos do modelo (sem intervenção)
2. Execução silenciosa das predições (equipe clínica não vê o resultado)
3. Follow-up de 30 dias para ascertainment do desfecho
4. Análise estatística: AUC, calibração (Brier score, curva de confiabilidade)
5. Análise de subgrupos: idade, gênero, diagnóstico, unidade de internação

**Critério de sucesso:** AUC ≥ 0.65 (IC 95% > 0.60)

### Fase 2 — Validação de Impacto Clínico (6–18 meses)
**Objetivo:** Demonstrar redução de readmissões evitáveis

**Procedimento (randomização por cluster):**
1. Unidades de internação randomizadas: intervenção (predição visível) vs. controle
2. Para pacientes de "Alto Risco": alerta para equipe + protocolo de transição de cuidados
3. Desfecho: taxa de readmissão em 30 dias por grupo

**Critério de sucesso:** Redução relativa ≥ 15% nas readmissões evitáveis

---

## 5. Checklist CEP — Documentos Necessários

- [ ] Protocolo de pesquisa (este documento + aprovação institucional)
- [ ] Termo de Consentimento Livre e Esclarecido (TCLE) ou justificativa de dispensa
- [ ] Declaração de confidencialidade e proteção de dados
- [ ] Currículo lattes dos pesquisadores principais
- [ ] Orçamento e cronograma
- [ ] Declaração de conflito de interesses
- [ ] Comprovante de infraestrutura (servidor, backup, acesso controlado)
- [ ] RIPD (Relatório de Impacto à Proteção de Dados) — ver `docs/lgpd_conformidade.md`
- [ ] Declaração do DPO sobre adequação à LGPD

---

## 6. Métricas de Go/No-Go para Produção

| Critério | Mínimo para Produção | Ideal |
|---|---|---|
| ROC-AUC (local) | ≥ 0.65 | ≥ 0.70 |
| Recall (alto risco) | ≥ 0.80 | ≥ 0.90 |
| Brier Score | ≤ 0.25 | ≤ 0.18 |
| AUC mínimo por subgrupo | ≥ 0.60 | ≥ 0.65 |
| Delta TPR entre gêneros | ≤ 0.20 | ≤ 0.10 |
| Aprovação CEP | Obrigatório | — |
| Treinamento equipe clínica | Obrigatório | — |

---

## 7. Referências

- Hanley JA, McNeil BJ. A method of comparing the areas under ROC curves derived from the same cases. *Radiology*, 148(3):839-43, 1983.
- ANVISA. RDC 657/2022 — Software como Dispositivo Médico.
- CFM. Resolução 2.227/2018 — Telemedicina e IA.
- LGPD. Lei 13.709/2018 — Lei Geral de Proteção de Dados Pessoais.
- Rajpurkar P et al. AI in health and medicine. *Nature Medicine*, 2022.
