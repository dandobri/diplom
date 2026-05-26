# Prompt sweep — comparison

Все варианты прогнаны на одном и том же:

- наборе clinical cases;

- одном и том же no-RAG ответе (общий baseline);

- одних и тех же judge promptах.


Единственное различие между строками — RAG prompt.


## Главное

| prompt_name | rag_judge_success_rate | rag_avg_claims_per_answer | rag_faithfulness_soft | rag_hallucination_rate | rag_citation_accuracy_rate | faithfulness_soft_improvement_abs | hallucination_rate_reduction_abs |
|---|---|---|---|---|---|---|---|
| rag_diagnostic_v1 | 1.0000 | 6.5500 | 0.8107 | 0.1489 | 0.8407 | 0.3695 | 0.1272 |
| rag_diagnostic_v2_strict | 1.0000 | 7.0000 | 0.7981 | 0.1446 | 0.8562 | 0.3675 | 0.2070 |
| rag_diagnostic_v3_compact | 1.0000 | 6.2000 | 0.7991 | 0.1675 | 0.8292 | 0.4234 | 0.2143 |
| rag_diagnostic_v2_strict | 1.0000 | 7.1000 | 0.7505 | 0.2246 | 0.7918 | 0.3317 | 0.1172 |

## Targets met

| prompt_name | rag_meets_faithfulness | rag_meets_hallucination | rag_meets_citation_accuracy | rag_meets_faithfulness_improvement | rag_meets_hallucination_reduction |
|---|---|---|---|---|---|
| rag_diagnostic_v1 | ✗ | ✓ | ✗ | ✓ | ✗ | (2/5) |
| rag_diagnostic_v2_strict | ✗ | ✓ | ✗ | ✓ | ✓ | (3/5) |
| rag_diagnostic_v3_compact | ✗ | ✗ | ✗ | ✓ | ✓ | (2/5) |
| rag_diagnostic_v2_strict | ✗ | ✗ | ✗ | ✓ | ✗ | (1/5) |
