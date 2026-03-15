# Conformidade com a LGPD

## Hospital Readmission Prediction API — Relatório de Adequação

**Versão:** 1.0 | **Data:** 2026-03-15 | **Lei:** Lei 13.709/2018 (LGPD)

---

## 1. Contexto Legal

Dados de saúde são **dados pessoais sensíveis** nos termos do Art. 5°, II da LGPD.
O tratamento de dados de saúde só é permitido nos casos do Art. 11, II:

> *"Art. 11. O tratamento de dados pessoais sensíveis somente poderá ocorrer nas seguintes hipóteses: [...] II - sem fornecimento de consentimento do titular, nas hipóteses em que for indispensável para: [...] f) tutela da saúde, exclusivamente, em procedimento realizado por profissionais de saúde, serviços de saúde ou autoridade sanitária;"*

**Base legal adotada:** Art. 11, II, f — tutela da saúde.

---

## 2. Mapeamento de Dados (Data Flow)

### 2.1 Dados que ENTRAM no sistema

| Campo | Tipo | Sensível? | Retenção |
|---|---|---|---|
| age_numeric | Numérico | Sim | Não armazenado |
| gender | Categórico | Sim | Não armazenado |
| diag_primary | Categórico | Sim | Não armazenado |
| time_in_hospital | Numérico | Sim | Não armazenado |
| num_medications, num_procedures, etc. | Numérico | Sim | Não armazenado |
| hba1c_result, glucose_serum_test | Categórico | Sim | Não armazenado |
| insulin, change_medications, etc. | Categórico | Sim | Não armazenado |

> ⚠ **Nenhum dado pessoal identificável (nome, CPF, prontuário, data de nascimento, endereço) é aceito ou armazenado pela API.**

### 2.2 Dados que SAEM do sistema

| Dado | Descrição | Sensível? |
|---|---|---|
| readmission_probability | Probabilidade 0–1 | Sim (decisão clínica) |
| risk_level | Baixo/Moderado/Alto | Sim |
| recommendation | Texto de recomendação | Não |
| feature_contributions | SHAP values | Sim |

### 2.3 Dados nos LOGS de Auditoria (banco SQLite)

| Campo | O que é | PII? |
|---|---|---|
| patient_hash | SHA-256 dos dados de entrada | Não (hash irreversível) |
| ip_address_hash | SHA-256 do IP | Não (hash irreversível) |
| user_id | Username do profissional | Sim (dado do sistema) |
| prediction, probability, risk_level | Resultado | Sim (dado de saúde derivado) |
| feature_hash | SHA-256 do vetor de features | Não |
| timestamp | Data/hora da predição | Não (sem link ao paciente) |

---

## 3. Papel do Encarregado de Dados (DPO)

Conforme Art. 41 da LGPD, organizações que realizam tratamento em larga escala de dados sensíveis devem indicar um Encarregado de Proteção de Dados (DPO).

**Responsabilidades do DPO neste contexto:**
- Receber e responder a solicitações dos titulares de dados
- Orientar a equipe sobre boas práticas de proteção de dados
- Supervisionar a conformidade deste sistema com a LGPD
- Comunicar incidentes à ANPD quando necessário (Art. 48)

---

## 4. RIPD — Relatório de Impacto à Proteção de Dados Pessoais

### 4.1 Necessidade do RIPD
O Art. 38 da LGPD exige RIPD quando o tratamento pode gerar riscos às liberdades civis e direitos fundamentais. **Este sistema requer RIPD** pois:
- Trata dados de saúde em escala
- Produz decisões com impacto potencial na assistência ao paciente
- Envolve tratamento automatizado (Art. 20 — direito à revisão)

### 4.2 Estrutura do RIPD (a ser preenchida pela instituição)

1. **Descrição do tratamento:** Predição de risco de readmissão hospitalar
2. **Finalidade:** Auxílio à decisão de alta hospitalar
3. **Base legal:** Art. 11, II, f
4. **Necessidade e proporcionalidade:** Apenas dados mínimos necessários
5. **Riscos identificados:** Viés algorítmico, decisões equivocadas
6. **Medidas de mitigação:** SHAP explicabilidade, análise de fairness, auditoria
7. **Avaliação final:** Risco residual aceitável com as medidas implementadas

---

## 5. Direitos dos Titulares (Art. 18)

Como **nenhum dado pessoal é armazenado com link direto ao paciente**, a maioria dos direitos do Art. 18 é satisfeita por design:

| Direito | Como é atendido |
|---|---|
| **Confirmação de tratamento** (Art. 18, I) | Sim — via endpoint GET /audit/summary |
| **Acesso** (Art. 18, II) | Limitado — dados são hasheados; titulares podem solicitar via DPO |
| **Correção** (Art. 18, III) | N/A — dados são hasheados e imutáveis |
| **Anonimização/bloqueio** (Art. 18, IV) | N/A — dados já são pseudonimizados |
| **Eliminação** (Art. 18, VI) | Via procedimento documentado (ver Seção 6) |
| **Portabilidade** (Art. 18, V) | Via GET /audit/export — apenas para o próprio usuário |
| **Informação sobre compartilhamento** (Art. 18, VII) | Dados não são compartilhados com terceiros |
| **Revisão de decisão automatizada** (Art. 20) | Predição é apenas auxílio — revisão pelo profissional de saúde |

---

## 6. Retenção e Eliminação de Dados

### 6.1 Política de Retenção
- Registros de auditoria: **7.300 dias (≈ 20 anos)** — baseado na obrigação de retenção de prontuários médicos (CFM Res. 1.821/2007: mínimo 20 anos)
- Campo `deletion_scheduled` marca a data prevista de eliminação

### 6.2 Procedimento de Eliminação
Quando um titular solicitar eliminação (através do DPO):
1. Identificar todos os registros associados ao `patient_hash` (gerado pelo solicitante informando seus dados)
2. Executar: `UPDATE prediction_log SET deletion_scheduled = '<hoje>' WHERE patient_hash = '<hash>'`
3. Processo de eliminação definitiva executado conforme política de retenção
4. Comunicar ao titular a conclusão

### 6.3 Incidentes de Segurança
Em caso de acesso não autorizado ao banco de auditoria:
- Notificar o DPO imediatamente
- Avaliar se houve exposição de dados de saúde derivados (probabilidades, risk_level)
- Comunicar à ANPD em até 72h se o incidente puder acarretar risco (Art. 48)

---

## 7. Medidas Técnicas de Segurança Implementadas

| Medida | Implementação |
|---|---|
| **Autenticação** | JWT com expiração configurável |
| **Controle de acesso** | Roles (admin, clinician, viewer) |
| **Pseudonimização** | SHA-256 dos dados de entrada e IP |
| **Rate limiting** | slowapi (30/min por IP em /predict) |
| **Integridade dos modelos** | SHA-256 dos artefatos verificado no startup |
| **Auditoria** | Log imutável de todas as predições |
| **HTTPS** | Configurar nginx como reverse proxy (ver deployment_guide.md) |
| **Sem PII nos logs** | Design por arquitetura — verificável no código `api/audit.py` |

---

## 8. Contatos

| Papel | Responsabilidade |
|---|---|
| **DPO (Encarregado)** | A ser nomeado pela instituição |
| **ANPD** | Autoridade Nacional de Proteção de Dados — www.gov.br/anpd |
| **Responsável técnico** | Ver contato no README do projeto |
