# LLM evaluation report

_Generated for output_dir=`runs/exp_full_with_llm_eval_v1/llm_eval`._

## 1. Experiment setup

| Поле | Значение |
|---|---|
| Started at | 2026-05-09T17:44:23.746052+00:00 |
| Finished at | 2026-05-09T18:41:28.423142+00:00 |
| Elapsed (sec) | 3424.6771 |
| Judge LLM provider | openai_compatible |
| Judge LLM provider type | openai |
| Judge LLM model | gpt-4o-mini |
| Faithfulness judge config | configs/prompts/judge_faithfulness_v1.yaml |
| Citation judge config | configs/prompts/judge_citation_accuracy_v1.yaml |
| Relevance judge config | configs/prompts/judge_answer_relevance_v1.yaml |
| Mode | both |
| Limit | — |
| Max claims per answer | 20 |
| Pairs input | 73 |
| Case evaluations total | 146 |

## 2. Datasets

- Pairs path: `runs/exp_full_with_llm_eval_v1/generation/rag_vs_no_rag_pairs.jsonl`
- Clinical cases path: `data/clinical_cases_v1.jsonl`
- Total cases evaluated: **73**

**Distribution by difficulty:**

| Difficulty | Count |
|---|---|
| medium | 33 |
| hard | 30 |
| easy | 10 |

**Distribution by case_type:**

| case_type | Count |
|---|---|
| diagnosis | 57 |
| diagnosis_and_next_steps | 9 |
| treatment | 7 |

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
| num_cases | 73 | 73 |
| valid_json_count | 73 | 73 |
| valid_json_rate | 1.0000 | 1.0000 |
| total_claims | 545 | 949 |
| supported_claims | 389 | 349 |
| partially_supported_claims | 30 | 89 |
| unsupported_claims | 98 | 322 |
| contradicted_claims | 0 | 0 |
| faithfulness_strict | 0.7253 | 0.3673 |
| faithfulness_soft | 0.7515 | 0.4167 |
| hallucination_rate | 0.1759 | 0.3220 |
| answer_relevance_avg | 4.9178 | 4.6986 |
| clinical_usefulness_avg | 4.9315 | 4.9315 |
| safety_avg | 5.0000 | 5.0000 |
| top1_diagnosis_match_rate | 0.8750 | 0.6250 |
| top3_diagnosis_match_rate | 0.8750 | 0.8750 |

**Citation metrics (RAG only):**

| Метрика | Значение |
|---|---|
| citation_coverage_item_level | 1.0000 |
| citation_coverage_claim_level | 0.9317 |
| citation_validity_rate | 1.0000 |
| citation_accuracy_rate | 0.7325 |
| invalid_citation_count_total | 0 |

**Comparison:**

| Метрика | Значение |
|---|---|
| faithfulness_soft_improvement_abs | 0.3348 |
| faithfulness_soft_improvement_rel | 0.8035 |
| hallucination_rate_reduction_abs | 0.1461 |
| case_level_wins_rag_better | 64 |
| case_level_losses_rag_worse | 7 |
| case_level_ties | 2 |

## 7. Where RAG > no-RAG / RAG < no-RAG (case-level)

**Top-5 cases where RAG > no-RAG (по Δ faithfulness_soft):**

| case_id | difficulty | RAG soft | no-RAG soft | Δ |
|---|---|---|---|---|
| q044 | easy | 1.0000 | 0.2083 | 0.7917 |
| h014 | hard | 1.0000 | 0.2143 | 0.7857 |
| q012 | easy | 0.7500 | 0.0000 | 0.7500 |
| q017 | medium | 1.0000 | 0.2500 | 0.7500 |
| q029 | hard | 1.0000 | 0.2500 | 0.7500 |

**Top-5 cases where RAG < no-RAG (по Δ faithfulness_soft):**

| case_id | difficulty | RAG soft | no-RAG soft | Δ |
|---|---|---|---|---|
| manual_002_acs_no_label | hard | 0.1111 | 0.3235 | -0.2124 |
| manual_008_elderly_vague | hard | 0.2857 | 0.4333 | -0.1476 |
| manual_007_hcm_young_syncope | hard | 0.5000 | 0.5769 | -0.0769 |
| q025 | medium | 0.7500 | 0.7778 | -0.0278 |
| h007 | hard | 0.2857 | 0.3125 | -0.0268 |

## 8. Target thresholds (ВКР)

| Цель | Значение | Pass/Fail |
|---|---|---|
| RAG faithfulness_soft >= 0.85 | 0.7515 | ✗ |
| RAG hallucination_rate < 0.15 | 0.1759 | ✗ |
| RAG citation_accuracy_rate >= 0.9 | 0.7325 | ✗ |
| Faithfulness_soft improvement (RAG − no-RAG) >= 0.2 | 0.3348 | ✓ |
| Hallucination_rate reduction (no-RAG − RAG) >= 0.2 | 0.1461 | ✗ |

**Итог: часть целей не достигнута. См. детальный анализ ошибок ниже.**

## 9. Failed judge cases

_Список пуст — все judge-вызовы успешны._

## 10. Error analysis

| Mode | total_claims | unsupported | contradicted |
|---|---|---|---|
| rag | 545 | 98 | 0 |
| no_rag | 949 | 322 | 0 |

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
