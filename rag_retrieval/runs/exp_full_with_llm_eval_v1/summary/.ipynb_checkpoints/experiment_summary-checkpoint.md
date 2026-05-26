# Experiment summary: `exp_full_with_llm_eval_v1`

- Status: **unknown**
- Description: Full RAG + LLM evaluation (faithfulness/hallucination/citation/relevance)
- Started: —
- Finished: —
- Total runtime, sec: —

## Experiment config

- output_dir: `runs/exp_full_with_llm_eval_v1`
- retrieval_queries_path: `data/retrieval_eval_queries_plus_hard_v1.jsonl`
- clinical_cases_path: `data/clinical_cases_v1.jsonl`
- prompt_config_rag: `configs/prompts/rag_diagnostic_v2_strict.yaml`
- prompt_config_no_rag: `configs/prompts/no_rag_diagnostic_v2_baseline.yaml`

## Retrieval results

_Skipped._

## Reranking / context-selection results

_Skipped._

## Generation results

- llm_provider: openai_compatible
- llm_model_name: gpt-4o-mini
- num_rag_answers: 73
- num_no_rag_answers: 73
- rag_valid_json_rate: 1.0000
- no_rag_valid_json_rate: 1.0000
- rag_avg_citation_count: 2.3288
- rag_avg_citation_coverage_estimate: 0.9932
- rag_invalid_citations_total: 0

## LLM evaluation results

| Метрика | RAG | no-RAG |
|---|---|---|
| faithfulness_soft | 0.7886 | 0.4304 |
| hallucination_rate | 0.1538 | 0.3485 |
| answer_relevance_avg | 4.8767 | 4.7260 |
| clinical_usefulness_avg | 4.8219 | 4.9315 |
| safety_avg | 5.0000 | 5.0000 |

**Comparison:**

- faithfulness_soft_improvement_abs: 0.3582
- hallucination_rate_reduction_abs: 0.1947
- case_level_wins_rag_better: 34
- case_level_losses_rag_worse: 0

**Targets met:**

- RAG faithfulness_soft >= 0.85: ✗
- RAG hallucination_rate < 0.15: ✗
- RAG citation_accuracy_rate >= 0.90: ✗
- Faithfulness improvement >= 0.20: ✓
- Hallucination reduction >= 0.20: ✗

Подробный отчёт: `llm_eval/llm_eval_report.md`

## Files produced

- `generation/.ipynb_checkpoints/no_rag_answers-checkpoint.jsonl`
- `generation/.ipynb_checkpoints/rag_answers-checkpoint.jsonl`
- `generation/.ipynb_checkpoints/rag_vs_no_rag_summary-checkpoint.csv`
- `generation/compare_rag_vs_no_rag.log`
- `generation/no_rag_answers.jsonl`
- `generation/no_rag_answers.jsonl.log`
- `generation/rag_answers.jsonl`
- `generation/rag_answers.jsonl.log`
- `generation/rag_vs_no_rag_pairs.jsonl`
- `generation/rag_vs_no_rag_summary.csv`
- `metrics/final_metrics_summary.csv`
- `metrics/final_metrics_summary.json`
- `metrics/final_metrics_summary.md`
- `metrics/generation_metrics.csv`
- `metrics/generation_metrics.json`

## Next steps

- Inspect detailed records: `rerank/detailed_results.jsonl` для retrieval-debug.
- Если метрики просели — обновить prompt-config или поменять reranker / context_selection.
