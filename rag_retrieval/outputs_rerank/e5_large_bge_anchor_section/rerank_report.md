# Reranking + context selection report

## Конфигурация

- Embedding model: `e5_large` (`intfloat/multilingual-e5-large`)
- Reranker: `bge_reranker_v2_m3` (`BAAI/bge-reranker-v2-m3`)
- context_selection: `anchor_section`
- candidate_top_k = 30, final_top_k = 5
- queries = 65

## Цели проекта (по финальному пайплайну)

- Recall@5 ≥ 0.80 — ✓ достигнут (final = 1.0000)
- Precision@5 ≥ 0.70 — ✓ достигнут (final = 0.7169)

## Сравнение dense → reranked → final (с context selection)

| Метрика | Dense top-5 | Reranked top-5 | Final top-5 | Δ (full) | Δ (CS only) |
|---|---|---|---|---|---|
| Document Hit@1 | 0.8000 | 0.9538 | 0.9538 | +0.1538 | +0.0000 |
| Document Hit@5 | 0.9846 | 1.0000 | 1.0000 | +0.0154 | +0.0000 |
| Document Recall@5 | 0.9846 | 1.0000 | 1.0000 | +0.0154 | +0.0000 |
| Document Precision@5 | 0.5754 | 0.6031 | 0.7169 | +0.1415 | +0.1138 |
| Document MRR | 0.8703 | 0.9744 | 0.9659 | +0.0956 | -0.0085 |
| Chunk Hit@5 | 0.9385 | 0.9385 | 0.9231 | -0.0154 | -0.0154 |
| Chunk Recall@5 | 0.9385 | 0.9385 | 0.9231 | -0.0154 | -0.0154 |
| Chunk Precision@5 | 0.1877 | 0.1877 | 0.1846 | -0.0031 | -0.0031 |
| Chunk MRR | 0.7708 | 0.9115 | 0.8992 | +0.1284 | -0.0123 |
| Section Hit@5 | 0.9846 | 1.0000 | 1.0000 | +0.0154 | +0.0000 |
| Section Precision@5 | 0.5046 | 0.5169 | 0.6554 | +0.1508 | +0.1385 |
| Section MRR | 0.8600 | 0.9744 | 0.9659 | +0.1059 | -0.0085 |
| Page Hit@1 | 0.7231 | 0.9077 | 0.9077 | +0.1846 | +0.0000 |
| Page Hit@5 | 0.9538 | 0.9846 | 0.9538 | +0.0000 | -0.0308 |
| Page Precision@5 | 0.3015 | 0.3200 | 0.3846 | +0.0831 | +0.0646 |
| Soft Page Hit@5 (±1) | 0.9538 | 0.9846 | 0.9692 | +0.0154 | -0.0154 |
| Label Hit@5 | 1.0000 | 0.9846 | 0.9538 | -0.0462 | -0.0308 |

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
| query embedding   | 119.40 |
| dense retrieval    | 1.11 |
| reranking          | 675.12 |
| context selection  | 3.35 |
| total              | 798.99 |

## Рекомендации

- Оба целевых порога достигнуты. Финальная связка — рабочая. Имеет смысл закрепить параметры пайплайна и зафиксировать reranker в RAG-сервисе.
- Сильное расхождение между document/section/page/chunk Hit@5 показывает, достаточно ли точно retrieval попадает на нужный фрагмент, а не только на нужный документ.
