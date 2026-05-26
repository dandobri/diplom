# Final metrics summary

- Status: **completed**
- Started: 2026-05-09T16:53:08.450999+00:00
- Finished: 2026-05-09T18:41:28.548263+00:00
- Total runtime, sec: 6500.097

## Retrieval / reranking

| Метрика | Значение |
|---|---|
| embedding_model | — |
| reranker | — |
| context_selection | — |
| retrieval_dataset | data/retrieval_eval_queries_plus_hard_v1.jsonl |
| number_of_retrieval_queries | — |
| dense_document_recall_at_5 | — |
| dense_document_precision_at_5 | — |
| reranked_document_recall_at_5 | — |
| reranked_document_precision_at_5 | — |
| final_document_recall_at_5 | — |
| final_document_precision_at_5 | — |
| final_chunk_hit_at_5 | — |
| final_section_hit_at_5 | — |
| final_page_hit_at_5 | — |

## Generation

| Метрика | Значение |
|---|---|
| llm_provider | openai_compatible |
| llm_model_name | gpt-4o-mini |
| prompt_config_rag | configs/prompts/rag_diagnostic_v2_strict.yaml |
| prompt_config_no_rag | configs/prompts/no_rag_diagnostic_v2_baseline.yaml |
| clinical_cases_path | data/clinical_cases_v1.jsonl |
| generation_limit | — |
| rag_num_answers | 73 |
| no_rag_num_answers | 73 |
| rag_valid_json_rate | 1.0000 |
| no_rag_valid_json_rate | 1.0000 |
| rag_avg_citation_count | 2.3425 |
| rag_invalid_citations_total | 0 |
| rag_avg_citation_coverage | 1.0000 |

## LLM evaluation

| Метрика | RAG | no-RAG |
|---|---|---|
| faithfulness_strict | 0.7253 | 0.3673 |
| faithfulness_soft | 0.7515 | 0.4167 |
| hallucination_rate | 0.1759 | 0.3220 |
| answer_relevance_avg | 4.9178 | 4.6986 |
| clinical_usefulness_avg | 4.9315 | 4.9315 |
| safety_avg | 5.0000 | 5.0000 |

**Citation metrics (RAG):**

| Метрика | Значение |
|---|---|
| rag_citation_coverage_claim_level | 0.9317 |
| rag_citation_coverage_item_level | 1.0000 |
| rag_citation_validity_rate | 1.0000 |
| rag_citation_accuracy_rate | 0.7325 |

**Comparison:**

| Метрика | Значение |
|---|---|
| faithfulness_soft_improvement_abs | 0.3348 |
| faithfulness_soft_improvement_rel | 0.8035 |
| hallucination_rate_reduction_abs | 0.1461 |
| case_level_wins_rag_better | 64 |
| case_level_losses_rag_worse | 7 |
| case_level_ties | 2 |

## Target thresholds (ВКР)

| Цель | Pass/Fail |
|---|---|
| RAG faithfulness_soft >= 0.85 | ✗ |
| RAG hallucination_rate < 0.15 | ✗ |
| RAG citation_accuracy_rate >= 0.90 | ✗ |
| Faithfulness improvement >= 0.20 | ✓ |
| Hallucination reduction >= 0.20 | ✗ |
