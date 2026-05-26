# Reranking + context selection report

## Конфигурация

- Embedding model: `e5_large` (`intfloat/multilingual-e5-large`)
- Reranker: `bge_reranker_v2_m3` (`BAAI/bge-reranker-v2-m3`)
- context_selection: `anchor_page`
- candidate_top_k = 30, final_top_k = 5
- queries = 65

## Цели проекта (по финальному пайплайну)

- Recall@5 ≥ 0.80 — ✓ достигнут (final = 0.9846)
- Precision@5 ≥ 0.70 — ✓ достигнут (final = 0.8431)

## Сравнение dense → reranked → final (с context selection)

| Метрика | Dense top-5 | Reranked top-5 | Final top-5 | Δ (full) | Δ (CS only) |
|---|---|---|---|---|---|
| Document Hit@1 | 0.8000 | 0.9538 | 0.9538 | +0.1538 | +0.0000 |
| Document Hit@5 | 0.9846 | 1.0000 | 0.9846 | +0.0000 | -0.0154 |
| Document Recall@5 | 0.9846 | 1.0000 | 0.9846 | +0.0000 | -0.0154 |
| Document Precision@5 | 0.5754 | 0.6031 | 0.8431 | +0.2677 | +0.2400 |
| Document MRR | 0.8703 | 0.9744 | 0.9608 | +0.0905 | -0.0136 |
| Chunk Hit@5 | 0.9385 | 0.9385 | 0.9231 | -0.0154 | -0.0154 |
| Chunk Recall@5 | 0.9385 | 0.9385 | 0.9231 | -0.0154 | -0.0154 |
| Chunk Precision@5 | 0.1877 | 0.1877 | 0.1846 | -0.0031 | -0.0031 |
| Chunk MRR | 0.7708 | 0.9115 | 0.8992 | +0.1284 | -0.0123 |
| Section Hit@5 | 0.9846 | 1.0000 | 0.9846 | +0.0000 | -0.0154 |
| Section Precision@5 | 0.5046 | 0.5169 | 0.6954 | +0.1908 | +0.1785 |
| Section MRR | 0.8600 | 0.9744 | 0.9608 | +0.1008 | -0.0136 |
| Page Hit@1 | 0.7231 | 0.9077 | 0.9077 | +0.1846 | +0.0000 |
| Page Hit@5 | 0.9538 | 0.9846 | 0.9231 | -0.0307 | -0.0615 |
| Page Precision@5 | 0.3015 | 0.3200 | 0.4554 | +0.1539 | +0.1354 |
| Soft Page Hit@5 (±1) | 0.9538 | 0.9846 | 0.9538 | +0.0000 | -0.0308 |
| Label Hit@5 | 1.0000 | 0.9846 | 0.9692 | -0.0308 | -0.0154 |

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
| query embedding   | 125.32 |
| dense retrieval    | 1.12 |
| reranking          | 670.36 |
| context selection  | 3.58 |
| total              | 800.37 |

## Рекомендации

- Оба целевых порога достигнуты. Финальная связка — рабочая. Имеет смысл закрепить параметры пайплайна и зафиксировать reranker в RAG-сервисе.
- Сильное расхождение между document/section/page/chunk Hit@5 показывает, достаточно ли точно retrieval попадает на нужный фрагмент, а не только на нужный документ.
