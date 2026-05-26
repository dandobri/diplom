# Reranking + context selection report

## Конфигурация

- Embedding model: `e5_large` (`intfloat/multilingual-e5-large`)
- Reranker: `bge_reranker_v2_m3` (`BAAI/bge-reranker-v2-m3`)
- context_selection: `anchor_document`
- candidate_top_k = 30, final_top_k = 5
- queries = 65

## Цели проекта (по финальному пайплайну)

- Recall@5 ≥ 0.80 — ✓ достигнут (final = 0.9538)
- Precision@5 ≥ 0.70 — ✓ достигнут (final = 0.9508)

## Сравнение dense → reranked → final (с context selection)

| Метрика | Dense top-5 | Reranked top-5 | Final top-5 | Δ (full) | Δ (CS only) |
|---|---|---|---|---|---|
| Document Hit@1 | 0.8000 | 0.9538 | 0.9538 | +0.1538 | +0.0000 |
| Document Hit@5 | 0.9846 | 1.0000 | 0.9538 | -0.0308 | -0.0462 |
| Document Recall@5 | 0.9846 | 1.0000 | 0.9538 | -0.0308 | -0.0462 |
| Document Precision@5 | 0.5754 | 0.6031 | 0.9508 | +0.3754 | +0.3477 |
| Document MRR | 0.8703 | 0.9744 | 0.9538 | +0.0835 | -0.0206 |
| Chunk Hit@5 | 0.9385 | 0.9385 | 0.9231 | -0.0154 | -0.0154 |
| Chunk Recall@5 | 0.9385 | 0.9385 | 0.9231 | -0.0154 | -0.0154 |
| Chunk Precision@5 | 0.1877 | 0.1877 | 0.1846 | -0.0031 | -0.0031 |
| Chunk MRR | 0.7708 | 0.9115 | 0.9038 | +0.1330 | -0.0077 |
| Section Hit@5 | 0.9846 | 1.0000 | 0.9538 | -0.0308 | -0.0462 |
| Section Precision@5 | 0.5046 | 0.5169 | 0.7262 | +0.2216 | +0.2093 |
| Section MRR | 0.8600 | 0.9744 | 0.9538 | +0.0938 | -0.0206 |
| Page Hit@1 | 0.7231 | 0.9077 | 0.9077 | +0.1846 | +0.0000 |
| Page Hit@5 | 0.9538 | 0.9846 | 0.9385 | -0.0153 | -0.0461 |
| Page Precision@5 | 0.3015 | 0.3200 | 0.3754 | +0.0739 | +0.0554 |
| Soft Page Hit@5 (±1) | 0.9538 | 0.9846 | 0.9385 | -0.0153 | -0.0461 |
| Label Hit@5 | 1.0000 | 0.9846 | 0.9846 | -0.0154 | +0.0000 |

## Покрытие разметкой (сколько запросов имеют разметку для каждой метрики)

| Метрика | Coverage |
|---|---|
| Document Hit@1 | 65 / 65 |
| Document Hit@5 | 65 / 65 |
| Document Recall@5 | 65 / 65 |
| Document Precision@5 | 65 / 65 |
| Document MRR | 65 / 65 |
| Chunk Hit@5 | 65 / 65 |
| Chunk Recall@5 | 65 / 65 |
| Chunk Precision@5 | 65 / 65 |
| Chunk MRR | 65 / 65 |
| Section Hit@5 | 65 / 65 |
| Section Precision@5 | 65 / 65 |
| Section MRR | 65 / 65 |
| Page Hit@1 | 65 / 65 |
| Page Hit@5 | 65 / 65 |
| Page Precision@5 | 65 / 65 |
| Soft Page Hit@5 (±1) | 65 / 65 |
| Label Hit@5 | 65 / 65 |

## Производительность

| Шаг | мс/запрос |
|---|---|
| query embedding   | 121.47 |
| dense retrieval    | 1.04 |
| reranking          | 669.81 |
| context selection  | 0.82 |
| total              | 793.14 |

## Рекомендации

- Оба целевых порога достигнуты. Финальная связка — рабочая. Имеет смысл закрепить параметры пайплайна и зафиксировать reranker в RAG-сервисе.
- Сильное расхождение между document/section/page/chunk Hit@5 показывает, достаточно ли точно retrieval попадает на нужный фрагмент, а не только на нужный документ.
