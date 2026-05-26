# API Benchmark Report

## Конфигурация

- URL: `http://localhost:8000/search`
- Запросы: `data/clinical_cases_v1.jsonl` (поле: `patient_case`, limit: 50)
- candidate_top_k: 30, final_top_k: 5
- use_reranker: True, context_selection: anchor_page
- concurrency: 1

## Результаты

| Метрика | Значение |
|---|---|
| Запросов всего | 50 |
| Успешных | 50 |
| Ошибок | 0 |
| Success rate | 100.0% |
| Avg latency (server) | 8791.9 мс |
| P50 latency (server) | 8723.8 мс |
| P95 latency (server) | 9655.6 мс |
| Max latency (server) | 10525.8 мс |
| Target latency | 2000 мс |
| Target pass rate | 0.0% |

## Вывод

⚠ Только 0.0% запросов уложились в целевое время 2000 мс. Возможно, нужно оптимизировать конфигурацию или уменьшить candidate_top_k.
