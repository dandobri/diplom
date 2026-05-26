# Reranking + context selection report

## Конфигурация

- Embedding model: `e5_large` (`intfloat/multilingual-e5-large`)
- Reranker: `bge_reranker_v2_m3` (`BAAI/bge-reranker-v2-m3`)
- context_selection: `none`
- candidate_top_k = 30, final_top_k = 5
- queries = 65

## Цели проекта (по финальному пайплайну)

- Recall@5 ≥ 0.80 — ✓ достигнут (final = 1.0000)
- Precision@5 ≥ 0.70 — ✗ не достигнут (final = 0.6031)

## Сравнение before / after reranking

| Метрика | Dense top-5 | Reranked top-5 | Δ |
|---|---|---|---|
| Document Hit@1 | 0.8000 | 0.9538 | +0.1538 |
| Document Hit@5 | 0.9846 | 1.0000 | +0.0154 |
| Document Recall@5 | 0.9846 | 1.0000 | +0.0154 |
| Document Precision@5 | 0.5754 | 0.6031 | +0.0277 |
| Document MRR | 0.8703 | 0.9744 | +0.1041 |
| Chunk Hit@5 | 0.9385 | 0.9385 | +0.0000 |
| Chunk Recall@5 | 0.9385 | 0.9385 | +0.0000 |
| Chunk Precision@5 | 0.1877 | 0.1877 | +0.0000 |
| Chunk MRR | 0.7708 | 0.9115 | +0.1407 |
| Section Hit@5 | 0.9846 | 1.0000 | +0.0154 |
| Section Precision@5 | 0.5046 | 0.5169 | +0.0123 |
| Section MRR | 0.8600 | 0.9744 | +0.1144 |
| Page Hit@1 | 0.7231 | 0.9077 | +0.1846 |
| Page Hit@5 | 0.9538 | 0.9846 | +0.0308 |
| Page Precision@5 | 0.3015 | 0.3200 | +0.0185 |
| Soft Page Hit@5 (±1) | 0.9538 | 0.9846 | +0.0308 |
| Label Hit@5 | 1.0000 | 0.9846 | -0.0154 |

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
| query embedding   | 132.90 |
| dense retrieval    | 1.02 |
| reranking          | 679.06 |
| context selection  | 0.04 |
| total              | 813.03 |

## Рекомендации

- Precision@5 ниже 0.70. Попробовать `--context-selection anchor_section` или `anchor_document` — это формирование финального RAG-контекста вокруг наиболее релевантного фрагмента.
- Сильное расхождение между document/section/page/chunk Hit@5 показывает, достаточно ли точно retrieval попадает на нужный фрагмент, а не только на нужный документ.
