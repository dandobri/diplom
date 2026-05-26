# LLM evaluation report

_Generated for output_dir=`runs/exp_promptsweep_v1/rag_diagnostic_v1/llm_eval`._

## 1. Experiment setup

| Поле | Значение |
|---|---|
| Started at | 2026-05-09T14:55:53.579901+00:00 |
| Finished at | 2026-05-09T15:10:19.614783+00:00 |
| Elapsed (sec) | 866.0349 |
| Judge LLM provider | openai_compatible |
| Judge LLM provider type | openai |
| Judge LLM model | gpt-4o-mini |
| Faithfulness judge config | configs/prompts/judge_faithfulness_v1.yaml |
| Citation judge config | configs/prompts/judge_citation_accuracy_v1.yaml |
| Relevance judge config | configs/prompts/judge_answer_relevance_v1.yaml |
| Mode | both |
| Limit | 20 |
| Max claims per answer | 20 |
| Pairs input | 20 |
| Case evaluations total | 40 |

## 2. Datasets

- Pairs path: `runs/exp_promptsweep_v1/rag_diagnostic_v1/generation/rag_vs_no_rag_pairs.jsonl`
- Clinical cases path: `data/clinical_cases_v1.jsonl`
- Total cases evaluated: **20**

**Distribution by difficulty:**

| Difficulty | Count |
|---|---|
| medium | 11 |
| hard | 5 |
| easy | 4 |

**Distribution by case_type:**

| case_type | Count |
|---|---|
| diagnosis | 19 |
| treatment | 1 |

## 3. RAG prompt

_Prompt config not loaded (path=None)._

## 4. no-RAG prompt

_Prompt config not loaded (path=None)._

## 5. Judge prompts

### 5.1. Faithfulness judge
- Path: `configs/prompts/judge_faithfulness_v1.yaml`
- Name: `judge_faithfulness_v1`
- Version: `1.0`
- Purpose: claim-level faithfulness / hallucination judge for RAG and no-RAG answers
- Rules:
  - Оценивать только по reference_context, без внешних знаний.
  - Сохранять claim_id и claim_text без изменений; длина и порядок claim_evaluations совпадают со входом.
  - support_status — одно из 5 фиксированных значений.
  - Для no_rag-mode citation-поля = null; для rag-mode — реальные true/false.
  - Ответ — строго валидный JSON, без markdown и без текста вне JSON.

### 5.2. Citation accuracy judge
- Path: `configs/prompts/judge_citation_accuracy_v1.yaml`
- Name: `judge_citation_accuracy_v1`
- Version: `1.0`
- Purpose: RAG citation-accuracy judge — checks each citation against the claim it supports
- Rules:
  - is_valid_reference — техническая проверка совпадения source_id/chunk_id/document_id/pages.
  - supports_claim — семантическая проверка: реально ли source_block_text подтверждает claim_text.
  - Если is_valid_reference == false → supports_claim == false.
  - Сохранять citation_id и порядок из входа.
  - Ответ — строго валидный JSON, без markdown и без текста вне JSON.

### 5.3. Answer relevance judge
- Path: `configs/prompts/judge_answer_relevance_v1.yaml`
- Name: `judge_answer_relevance_v1`
- Version: `1.0`
- Purpose: answer-level relevance / clinical usefulness / safety judge for RAG and no-RAG
- Rules:
  - Все три шкалы — целые числа от 1 до 5.
  - has_disclaimer и states_final_diagnosis — булевы.
  - Не сравнивать с альтернативным режимом (RAG vs no_rag) — судим только данный ответ.
  - Ответ — строго валидный JSON, без markdown и без текста вне JSON.

## 6. Main metrics (RAG vs no-RAG)

| Метрика | RAG | no-RAG |
|---|---|---|
| num_cases | 20 | 20 |
| valid_json_count | 20 | 20 |
| valid_json_rate | 1.0000 | 1.0000 |
| total_claims | 131 | 217 |
| supported_claims | 101 | 80 |
| partially_supported_claims | 6 | 27 |
| unsupported_claims | 21 | 66 |
| contradicted_claims | 0 | 0 |
| faithfulness_strict | 0.7890 | 0.3737 |
| faithfulness_soft | 0.8107 | 0.4412 |
| hallucination_rate | 0.1489 | 0.2761 |
| answer_relevance_avg | 4.9000 | 4.6500 |
| clinical_usefulness_avg | 4.9500 | 4.9000 |
| safety_avg | 5.0000 | 5.0000 |
| top1_diagnosis_match_rate | — | — |
| top3_diagnosis_match_rate | — | — |

**Citation metrics (RAG only):**

| Метрика | Значение |
|---|---|
| citation_coverage_item_level | 1.0000 |
| citation_coverage_claim_level | 0.9685 |
| citation_validity_rate | 1.0000 |
| citation_accuracy_rate | 0.8407 |
| invalid_citation_count_total | 0 |

**Comparison:**

| Метрика | Значение |
|---|---|
| faithfulness_soft_improvement_abs | 0.3695 |
| faithfulness_soft_improvement_rel | 0.8375 |
| hallucination_rate_reduction_abs | 0.1272 |
| case_level_wins_rag_better | 20 |
| case_level_losses_rag_worse | 0 |
| case_level_ties | 0 |

## 7. Where RAG > no-RAG / RAG < no-RAG (case-level)

**Top-5 cases where RAG > no-RAG (по Δ faithfulness_soft):**

| case_id | difficulty | RAG soft | no-RAG soft | Δ |
|---|---|---|---|---|
| q012 | easy | 1.0000 | 0.0556 | 0.9444 |
| q015 | hard | 1.0000 | 0.3125 | 0.6875 |
| q017 | medium | 0.9000 | 0.2500 | 0.6500 |
| q014 | easy | 1.0000 | 0.4231 | 0.5769 |
| q006 | easy | 0.8000 | 0.2500 | 0.5500 |

**Top-5 cases where RAG < no-RAG (по Δ faithfulness_soft):**

| case_id | difficulty | RAG soft | no-RAG soft | Δ |
|---|---|---|---|---|
| q009 | hard | 0.7143 | 0.6538 | 0.0605 |
| q011 | hard | 0.6875 | 0.6250 | 0.0625 |
| q019 | easy | 0.7143 | 0.6250 | 0.0893 |
| q005 | medium | 0.3750 | 0.2727 | 0.1023 |
| q018 | hard | 0.5714 | 0.3824 | 0.1890 |

## 8. Target thresholds (ВКР)

| Цель | Значение | Pass/Fail |
|---|---|---|
| RAG faithfulness_soft >= 0.85 | 0.8107 | ✗ |
| RAG hallucination_rate < 0.15 | 0.1489 | ✓ |
| RAG citation_accuracy_rate >= 0.9 | 0.8407 | ✗ |
| Faithfulness_soft improvement (RAG − no-RAG) >= 0.2 | 0.3695 | ✓ |
| Hallucination_rate reduction (no-RAG − RAG) >= 0.2 | 0.1272 | ✗ |

**Итог: часть целей не достигнута. См. детальный анализ ошибок ниже.**

## 9. Failed judge cases

_Список пуст — все judge-вызовы успешны._

## 10. Error analysis

| Mode | total_claims | unsupported | contradicted |
|---|---|---|---|
| rag | 131 | 21 | 0 |
| no_rag | 217 | 66 | 0 |

## 11. Limitations

- Это **LLM-as-a-judge** оценка. Она НЕ заменяет ручную экспертно-врачебную проверку.
- Часть кейсов в `data/clinical_cases_v1.jsonl` помечены как `auto_generated`
  (review_status) и нуждаются в ручной верификации формулировок и
  `expected_diagnoses`.
- `citation_validity_rate` — техническая метрика (совпадение source_id /
  chunk_id / document_id с retrieved_chunks). Семантическое подтверждение
  (`citation_accuracy_rate`) даётся judge-моделью.
- `faithfulness_*` и `hallucination_rate` зависят от качества reference_context.
  Для no-RAG в качестве reference берётся набор retrieved_chunks из парного
  RAG-запуска (apples-to-apples), но это не идеальный ground truth — часть
  правильных no-RAG-claims может быть помечена как `not_enough_information`,
  если они не попали в retrieval.
- Финальная экспертно-врачебная проверка ответов рекомендуется для
  пограничных или жизнеугрожающих сценариев.
