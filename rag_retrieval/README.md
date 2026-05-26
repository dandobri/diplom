# RAG Retrieval Pipeline (медицинские клинические рекомендации)

Этот проект реализует **retrieval-слой** RAG-модуля диагностической системы. На этом этапе **нет** генерации диагнозов, цитирования, faithfulness, Telegram-бота и REST-API — только подготовка и сравнение embedding-моделей и подготовка экспорта в `pgvector`.

## Experiment runner (один YAML — один прогон)

Чтобы менять только данные / промпты / параметры и запускать весь pipeline одной командой, используется experiment-orchestrator.

```bash
python -m src.run_experiment \
  --config configs/experiments/exp_full_v1.yaml
```

Что происходит:

1. retrieval evaluation (`evaluate_models`)
2. reranking + context selection (`evaluate_with_reranker`)
3. RAG generation (`generate_answers --mode rag`)
4. no-RAG generation (`generate_answers --mode no_rag`)
5. RAG vs no-RAG comparison (`compare_rag_vs_no_rag`)
6. сборка `summary/` и `metrics/final_metrics_summary.{json,csv,md}` (`collect_experiment_summary`)

Каждый этап опционален — выключается полем `enabled: false` в конфиге.

CLI overrides:

```text
--output-dir <dir>      # переопределить experiment.output_dir
--limit N               # переопределить generation.limit
--device cuda|cpu|auto  # переопределить device
--overwrite             # разрешить запись поверх непустого output_dir
--skip-retrieval --skip-rerank --skip-generation --skip-comparison
```

### Что меняется без правки кода

| Что | Где |
|---|---|
| retrieval-запросы (eval) | `data.retrieval_queries_path` в experiment YAML |
| clinical cases для LLM | `data.clinical_cases_path` + `generation.mode_cases_field.{case_id_field, patient_case_field}` |
| RAG prompt | `generation.prompt_config_rag` → `configs/prompts/*.yaml` |
| no-RAG prompt | `generation.prompt_config_no_rag` |
| embedding-модели для retrieval | `retrieval_eval.model_keys` |
| reranker / context selection | `reranking_eval.{reranker_key, context_selection}` |
| LLM-провайдер / модель | `generation.{llm_provider, llm_model}` (или env через `configs/llm.yaml`) |

### Готовые experiment-конфиги

- `configs/experiments/exp_full_v1.yaml` — полный прогон (retrieval + rerank + RAG/no-RAG + сравнение)
- `configs/experiments/exp_retrieval_only.yaml` — только retrieval + reranking, без LLM
- `configs/experiments/exp_llm_smoke.yaml` — только LLM-генерация, `limit=5` для проверки prompt'а

### Структура `runs/<experiment>/`

```
runs/<experiment>/
  config_resolved.yaml
  experiment.log
  datasphere_pipeline_commands.md         # команды для копирования в DataSphere
  retrieval/                              # evaluate_models output
    reports/{embedding_model_comparison.csv,.json, best_embedding_model.json}
    retrieval_results/<key>_detailed_results.jsonl
  rerank/                                 # evaluate_with_reranker output
    detailed_results.jsonl
    rerank_comparison_metrics.{csv,json}
    rerank_report.md
  generation/
    rag_answers.jsonl
    no_rag_answers.jsonl
    rag_vs_no_rag_pairs.jsonl
    rag_vs_no_rag_summary.csv
  metrics/
    pipeline_status.json                  # обновляется после каждого этапа
    retrieval_metrics.{json,csv}
    rerank_metrics.{json,csv}
    generation_metrics.{json,csv}
    final_metrics_summary.{json,csv,md}
  summary/
    experiment_summary.{json,md}
```

### DataSphere commands

Команды для последовательного исполнения в Yandex DataSphere генерируются автоматически в `runs/<experiment>/datasphere_pipeline_commands.md`. Шаблон — `configs/experiments/datasphere_pipeline_commands_template.md`. Можно перегенерировать вручную:

```bash
python -m src.write_datasphere_commands \
  --config configs/experiments/exp_full_v1.yaml \
  --output runs/exp_full_v1/datasphere_pipeline_commands.md
```

### CLI отдельных этапов (продолжают работать как раньше)

`generate_answers.py` теперь поддерживает кастомные prompt-конфиги и поля JSONL:

```bash
python -m src.generate_answers \
  --cases-path data/retrieval_eval_queries_plus_hard_v1.jsonl \
  --case-id-field query_id --patient-case-field query \
  --mode rag \
  --llm-config configs/llm.yaml \
  --prompt-config configs/prompts/rag_diagnostic_v1.yaml \
  --embedding-model-key e5_large \
  --embedding-config configs/embedding_models.yaml \
  --embeddings-dir outputs/embeddings/e5_large \
  --reranker-key bge_reranker_v2_m3 \
  --reranker-config configs/rerankers.yaml \
  --output-path runs/manual/rag_answers.jsonl --overwrite
```

---

## Структура проекта

```
rag_retrieval/
├── configs/
│   └── embedding_models.yaml        # описание embedding-моделей
├── data/
│   ├── chunks_v1.jsonl              # подготовленные чанки
│   └── retrieval_eval_queries_v1.jsonl
├── outputs/
│   ├── embeddings/<model_key>/      # один артефакт = одна модель
│   ├── retrieval_results/           # detailed_results.jsonl на модель
│   ├── reports/                     # сравнение и best_embedding_model.json
│   └── pgvector/                    # экспорт под PostgreSQL + pgvector
├── src/
│   ├── io_utils.py
│   ├── validate_chunks.py
│   ├── embedding_models.py
│   ├── build_embeddings.py
│   ├── retrieval.py
│   ├── metrics.py
│   ├── evaluate_models.py
│   ├── report.py
│   └── pgvector_export.py
├── notebooks/
│   ├── 01_validate_chunks.ipynb
│   ├── 02_build_embeddings_one_model.ipynb
│   └── 03_compare_embedding_models.ipynb
├── requirements.txt
└── README.md
```

Все CLI-скрипты запускаются как модули из корня проекта (там, где находится папка `src/`).

---

## 1. Установка

```bash
cd rag_retrieval
pip install -r requirements.txt
```

На Yandex DataSphere обычно достаточно:

```bash
pip install -r requirements.txt
```

Если в окружении нет CUDA-сборки PyTorch — поставьте подходящую сборку с pytorch.org вручную (в DataSphere GPU-окружении уже стоит).

---

## 2. Проверка чанков

Запускается один раз перед embeddings.

```bash
python -m src.validate_chunks \
  --chunks-path data/chunks_v1.jsonl \
  --output-dir outputs/reports
```

Что проверяется:
- обязательные поля (`id`, `document_id`, `text`, `embedding_text`, ...);
- непустые `text` и `embedding_text`;
- дубли `id` и `content_hash`;
- агрегированная статистика по `label`, `specialty`, `document_id`;
- средняя длина текста.

Отчёт сохраняется в:
- `outputs/reports/chunk_validation_report.json`
- `outputs/reports/chunk_validation_report.csv`

---

## 3. Parallel embedding generation in Yandex DataSphere

Идея: одна модель = один независимый запуск `build_embeddings.py` = одна папка с артефактами. Это позволяет считать разные модели **параллельно** в нескольких ноутбуках с GPU.

### Что подготовить один раз

В каждый ноутбук загрузить общий архив проекта:
- `src/`
- `configs/embedding_models.yaml`
- `data/chunks_v1.jsonl`
- `data/retrieval_eval_queries_v1.jsonl`
- `requirements.txt`

В каждом ноутбуке:

```bash
pip install -r requirements.txt
```

### Запуск моделей (по одной на ноутбук)

**Ноутбук 1 — multilingual-e5-large:**
```bash
python -m src.build_embeddings \
  --model-key e5_large \
  --chunks-path data/chunks_v1.jsonl \
  --config configs/embedding_models.yaml \
  --output-dir outputs/embeddings/e5_large \
  --device cuda \
  --overwrite
```

**Ноутбук 2 — BGE-M3:**
```bash
python -m src.build_embeddings \
  --model-key bge_m3 \
  --chunks-path data/chunks_v1.jsonl \
  --config configs/embedding_models.yaml \
  --output-dir outputs/embeddings/bge_m3 \
  --device cuda \
  --overwrite
```

**Ноутбук 3 — paraphrase-multilingual-mpnet:**
```bash
python -m src.build_embeddings \
  --model-key mpnet_multilingual \
  --chunks-path data/chunks_v1.jsonl \
  --config configs/embedding_models.yaml \
  --output-dir outputs/embeddings/mpnet_multilingual \
  --device cuda \
  --overwrite
```

**Ноутбук 4 (опционально) — multilingual-e5-large-instruct:**
```bash
python -m src.build_embeddings \
  --model-key e5_large_instruct \
  --chunks-path data/chunks_v1.jsonl \
  --config configs/embedding_models.yaml \
  --output-dir outputs/embeddings/e5_large_instruct \
  --device cuda \
  --overwrite
```

### Доступные флаги

| Флаг | Назначение |
|---|---|
| `--model-key` | ключ модели из `embedding_models.yaml` |
| `--chunks-path` | путь к `chunks_v1.jsonl` |
| `--config` | путь к YAML-конфигу |
| `--output-dir` | папка артефактов (одна на модель!) |
| `--device` | `auto` / `cuda` / `cpu` |
| `--batch-size` | переопределить `batch_size` из конфига |
| `--limit` | для отладки, обрабатывать первые N чанков |
| `--overwrite` | разрешить перезапись готовой папки |
| `--resume` | best-effort: использовать готовый артефакт, если совместим |
| `--trust-remote-code` | принудительно `trust_remote_code=True` |

### Что появится в `outputs/embeddings/<model_key>/`

```
embeddings.npy        # (N, D) float32, порядок строго совпадает с metadata.jsonl
metadata.jsonl        # все исходные поля чанка для каждой строки
run_info.json         # model_key, model_name, embedding_dim, device, gpu_name, тайминги
errors.jsonl          # (опционально) пропущенные чанки
```

### После завершения

Скачать или объединить папки `outputs/embeddings/*` в один проект, где будет запускаться evaluate.

### Логи в Yandex DataSphere

При запуске через `!python -m src.build_embeddings ...` Jupyter в DataSphere буферизует stdout subprocess'а — кажется, что «ничего не происходит», хотя на самом деле модель грузится / считается. Что есть и что делать:

1. **Лог-файл** дублируется в `outputs/embeddings/<model_key>/build_embeddings.log` и `outputs/evaluate_models.log`. В соседней ячейке можно смотреть его в реальном времени:
   ```bash
   !tail -f outputs/embeddings/e5_large/build_embeddings.log
   ```
2. **Запускать без буферизации** — добавьте `-u`:
   ```bash
   !python -u -m src.build_embeddings --model-key e5_large ...
   ```
3. **Самый надёжный способ — вызывать функцию из ячейки**, без subprocess. Тогда tqdm и логи показываются мгновенно. Шаблон лежит в `notebooks/02_build_embeddings_one_model.ipynb`:
   ```python
   from src.build_embeddings import build_embeddings_for_model
   run_info = build_embeddings_for_model(
       model_key='e5_large',
       chunks_path='docs/minzdrav-parsed',
       config_path='configs/embedding_models.yaml',
       output_dir='outputs/embeddings/e5_large',
       device='cuda',
       overwrite=True,
   )
   ```

---

## 3.1. Генерация retrieval evaluation dataset

Чтобы оценить retrieval, нужен размеченный набор запросов `data/retrieval_eval_queries_v1.jsonl`. Готовить его полностью руками долго, поэтому есть полуавтоматический генератор: он отбирает кандидатов из реальных чанков, фильтрует мусор, группирует по разделам и пишет черновики запросов, которые затем нужно проверить руками.

```bash
python -m src.generate_retrieval_eval_queries \
  --chunks-path docs/minzdrav-parsed \
  --output-path data/retrieval_eval_queries_v1.jsonl \
  --candidates-path outputs/reports/retrieval_eval_candidates.csv \
  --num-queries 50 \
  --max-queries-per-document 3 \
  --min-documents 15 \
  --seed 42
```

Что произойдёт:
1. Загружаются все чанки.
2. Отфильтровываются: слишком короткие тексты, оглавления, списки литературы, приложения.
3. Чанки группируются по `(document_id, section_id, label)` — одна группа = один потенциальный запрос с `relevance_level = "section"`.
4. Каждой группе ставится эвристическое `quality_score`.
5. **`outputs/reports/retrieval_eval_candidates.csv`** — все кандидаты, отсортированные по качеству. Полезно открыть глазами и понять, что за разделы у вас в корпусе.
6. Стратифицированный отбор: соблюдаются доли по `query_type`, `min_documents` уникальных документов, `max_queries_per_document` на документ.
7. По шаблонам генерируются черновики запросов разной сложности (`easy ≈ 30 %`, `medium ≈ 50 %`, `hard ≈ 20 %`).
8. Файл сохраняется как `data/retrieval_eval_queries_v1.jsonl`. Все записи помечены:
   ```json
   "review_status": "auto_generated",
   "requiring_human_review": true
   ```

### Шаблон одной записи

См. `data/retrieval_eval_queries.example.jsonl` — там 13 примеров со всеми полями, включая `differential_diagnosis` через два документа и пациентские формулировки уровня `hard`. Это эталон того, как должна выглядеть `verified` запись.

### Manual review (обязательно перед публикацией метрик)

Авто-черновики годятся как **стартовая точка**, но для честного сравнения моделей нужен ручной проход. Открывайте `data/retrieval_eval_queries_v1.jsonl` (например, в JupyterLab или VSCode), и для каждого запроса:

1. **Прочитать `query`** и `source_evidence[*].evidence_text_preview`. Если запрос звучит криво, переформулируйте — он должен быть похож на то, что спросил бы врач или пациент. Это самое важное.
2. **Проверить `expected_document_ids`**. По смыслу запроса откройте chunk и убедитесь, что именно этот документ вы хотите видеть в топе. Если релевантных документов несколько (например, для дифф. диагностики) — допишите.
3. **Проверить `expected_chunk_ids`**. Если в группе несколько чанков и все они отвечают на запрос — оставьте как есть; если ровно один — оставьте только его и поставьте `relevance_level = "chunk"`.
4. **`expected_section_keywords`** — это «sanity-check» для секционной метрики. Уберите служебные слова, добавьте реальные клинические термины (названия препаратов, методов, состояний).
5. **`difficulty`** — поправьте, если черновик не соответствует:
   - `easy` — прямой запрос с явным названием состояния;
   - `medium` — описательная формулировка, парафраз;
   - `hard` — синонимы, сокращения, пациентские слова, неполный контекст.
6. **`query_type`** — поправьте, если автомат угадал не тот label.
7. После проверки **поменять**:
   ```json
   "review_status": "verified",
   "requiring_human_review": false
   ```

### Сколько запросов реально нужно

- `--num-queries 30` — минимум для дипломной защиты, метрики ещё шумные.
- `--num-queries 50` — рекомендуемое значение, ~1.5–2 % погрешности на Recall@5.
- Покрытие: целевое `min_documents = 15` гарантирует, что одна модель не выиграет «случайно» на одном документе.

### Что писать в дипломе про методику

> Тестовый набор сформирован полуавтоматически: автоматический pipeline отобрал кандидаты из чанков клинических рекомендаций по эвристикам качества (длина текста, тип раздела, наличие label), сгруппировал чанки по разделам и сгенерировал черновики запросов. Все запросы прошли ручную верификацию автором: формулировка, ожидаемые документы и ключевые слова разделов вычитаны и при необходимости переписаны. Размеченный набор содержит N запросов на M документах, со стратификацией по типу (treatment / diagnosis / symptoms / lab_tests / instrumental_tests / risk_factors / contraindications / follow_up / differential_diagnosis / emergency / general) и по сложности (easy / medium / hard ≈ 30/50/20 %).

## 4. Сравнение моделей

После того как все параллельные запуски завершились и папки собраны в один проект:

```bash
python -m src.evaluate_models \
  --queries-path data/retrieval_eval_queries_v1.jsonl \
  --config configs/embedding_models.yaml \
  --embeddings-root outputs/embeddings \
  --model-keys e5_large bge_m3 mpnet_multilingual \
  --output-dir outputs
```

Что делает `evaluate_models.py`:
1. **Не пересчитывает** document embeddings — читает готовые `embeddings.npy` + `metadata.jsonl` + `run_info.json`.
2. Проверяет совместимость артефактов (число строк, embedding_dim, model_key).
3. Загружает запросы из `data/retrieval_eval_queries_v1.jsonl`.
4. Той же моделью считает **только query embeddings** (с правильным префиксом / instruction).
5. Делает retrieval top-10 (numpy dot product по нормализованным векторам).
6. Считает Hit@1/5/10, Recall@5, Precision@5, MRR, опционально section-keyword Hit@5.
7. Сохраняет:
   - `outputs/retrieval_results/<model_key>_detailed_results.jsonl`
   - `outputs/reports/embedding_model_comparison.csv`
   - `outputs/reports/embedding_model_comparison.json`
   - `outputs/reports/best_embedding_model.json`

---

## 5. Просмотр результатов

- **`outputs/reports/embedding_model_comparison.csv`** — таблица «модель × метрика» (Recall@5, Precision@5, MRR, Hit@k, среднее время).
- **`outputs/reports/best_embedding_model.json`** — победитель + причина выбора.
- **`outputs/retrieval_results/<model_key>_detailed_results.jsonl`** — построчные результаты для разбора ошибок.

### Как интерпретировать метрики

- **Recall@5** — главная цель этапа (≥ 0.80). Если у нас в топ-5 нет ни одного релевантного документа, RAG-pipeline не сработает.
- **Precision@5** — насколько шум в топ-5 невелик. Цель ≥ 0.70. На малом числе чанков легко получить «всё лишнее».
- **Hit@1** — сильный, но строгий сигнал «модель сразу понимает запрос». Полезен для ранжирования моделей при сравнимом recall.
- **MRR** — компромисс: учитывает, на каком ранге вообще нашёлся релевантный результат.
- **avg_query_time_ms** — время кодирования одного запроса этой моделью.
- **avg_retrieval_time_ms** — время самого numpy-поиска. По требованию проекта на 5000 чанков должно быть ≤ 2000 мс — обычно это ~1–10 мс на численном numpy-поиске.

### Логика выбора лучшей модели

1. максимальный `recall_at_5`
2. при равенстве — максимальный `mrr`
3. при равенстве — максимальный `precision_at_5`
4. при равенстве — минимальный `avg_retrieval_time_ms`

---

## 5.1. Reranking evaluation (этап 2)

Когда лучшая dense-модель выбрана, добавляем reranker, чтобы поднять `Precision@5` и проверить, попадает ли система не просто в нужный документ, но в нужный **раздел / страницу / чанк**.

Pipeline:

```
query ──► dense top-N (30) ──► reranker scoring ──► top-K (5) ──► метрики
```

Скрипт: `src/evaluate_with_reranker.py`. Reranker по умолчанию — `BAAI/bge-reranker-v2-m3` через `sentence-transformers CrossEncoder` (см. `configs/rerankers.yaml`). Альтернативный backend `flag_embedding` доступен, если установлен `pip install FlagEmbedding`.

### Что добавляет этот этап к метрикам

В `metrics.py` метрики считаются на уровне `document_id`. В `advanced_metrics.py` — четыре уровня:

| Уровень | Что считается релевантным |
|---|---|
| **document** | `retrieved.document_id ∈ expected_document_ids` |
| **chunk** | `retrieved.id ∈ expected_chunk_ids` |
| **section** | document релевантен И (`section_id ∈ expected_section_ids` OR `section_title ∈ expected_section_titles` OR хотя бы одно `expected_section_keyword` встречается в section_title/text) |
| **page** | `retrieved.document_id == evidence.document_id` И диапазоны страниц пересекаются (`soft_*` — допускает ±1 страницу) |

Если у запроса **нет соответствующей разметки** (например, нет `expected_chunk_ids` или нет `source_evidence` с `page_start`), метрика для этого запроса возвращает `None` и **пропускается при усреднении**, а не штрафует модель нулём. В отчёте видно `coverage = N / total` — сколько запросов реально оцениваются.

### Команды запуска

**1. e5_large + bge reranker на основном наборе:**
```bash
python -m src.evaluate_with_reranker \
  --queries-path data/retrieval_eval_queries_plus_hard_v1.jsonl \
  --embedding-model-key e5_large \
  --embedding-config configs/embedding_models.yaml \
  --embeddings-dir outputs/embeddings/e5_large \
  --reranker-key bge_reranker_v2_m3 \
  --reranker-config configs/rerankers.yaml \
  --candidate-top-k 30 \
  --final-top-k 5 \
  --device cuda \
  --output-dir outputs_rerank/e5_large_bge_reranker
```

**2. e5_large_instruct + bge reranker на hard-only наборе:**
```bash
python -m src.evaluate_with_reranker \
  --queries-path data/retrieval_eval_queries_hard_v1.jsonl \
  --embedding-model-key e5_large_instruct \
  --embedding-config configs/embedding_models.yaml \
  --embeddings-dir outputs/embeddings/e5_large_instruct \
  --reranker-key bge_reranker_v2_m3 \
  --reranker-config configs/rerankers.yaml \
  --candidate-top-k 30 \
  --final-top-k 5 \
  --device cuda \
  --output-dir outputs_rerank/e5_large_instruct_hard_bge_reranker
```

**3. Быстрая отладка на CPU (3 запроса):**
```bash
python -m src.evaluate_with_reranker \
  --queries-path data/retrieval_eval_queries_hard_v1.jsonl \
  --embedding-model-key e5_large_instruct \
  --embedding-config configs/embedding_models.yaml \
  --embeddings-dir outputs/embeddings/e5_large_instruct \
  --reranker-key bge_reranker_v2_m3 \
  --reranker-config configs/rerankers.yaml \
  --candidate-top-k 10 \
  --final-top-k 5 \
  --device cpu \
  --limit 3 \
  --output-dir outputs_rerank/debug \
  --overwrite
```

Флаги:

| Флаг | Назначение |
|---|---|
| `--candidate-top-k` | Сколько кандидатов брать в dense-этапе (по умолчанию 30). Чем больше — тем выше потенциальный recall reranking, но дольше. |
| `--final-top-k` | Размер итоговой выдачи и k для всех Hit@k/Precision@k/Recall@k (по умолчанию 5). |
| `--save-all-candidates` | Сохранить в `detailed_results.jsonl` все 30 кандидатов, а не только top-5. Полезно для анализа того, что отсёк reranker. |
| `--limit N` | Прогон только на первых N запросах — для быстрой проверки перед полным запуском. |
| `--overwrite` | Перезаписать готовый `detailed_results.jsonl` в `--output-dir`. |

### Что появится в `--output-dir`

```
detailed_results.jsonl              # на каждый запрос: dense_top_results, reranked_top_results,
                                    #   dense_metrics, reranked_metrics, timings
rerank_comparison_metrics.csv       # одна строка: dense_*, reranked_*, delta_*, timing
rerank_comparison_metrics.json      # то же + конфиг + per-key coverage
rerank_report.md                    # читаемый отчёт для ВКР: таблица before/after, coverage,
                                    #   достижение целей, рекомендации
evaluate_with_reranker.log          # логи запуска (можно tail -f)
```

### Как интерпретировать результаты

1. **Document Recall@5** — потолок reranker'а. Если в dense-этапе нужный документ не попал в top-N, никакой reranker его уже не вытащит. Если `dense_document_recall_at_5 < 0.80` — увеличивайте `--candidate-top-k` до 50–60.
2. **Document Precision@5** — обычно главное, что улучшает reranker. Целевое: ≥ 0.70. Если рост маленький, вероятно top-N уже почти весь релевантен (мало шума, нечего отсеивать) — растёт меньше.
3. **Chunk / Page / Section Hit@5** — диагностические. Сильный рост на section_hit при том же document_recall означает, что reranker действительно умеет различать «правильный раздел» внутри документа.
4. **MRR** — насколько релевантный результат поднялся ближе к первому месту.
5. **`coverage` в JSON** — если `chunk_hit_at_5_coverage = 12/65`, значит у 53 запросов нет `expected_chunk_ids`, и chunk-level метрика по сути считается на 12 запросах. Для ВКР про эту метрику стоит писать как «по подмножеству N запросов с chunk-level разметкой».

### Сколько это работает

| Сценарий | candidate=30, queries=65, top_k=5 |
|---|---|
| GPU (T4 / A10) | ~30–60 сек на запросы + загрузка reranker (~30 сек первый раз) |
| CPU | ~3–5 сек на запрос (≈3–5 мин на 65 запросов) |
| Скачивание `BAAI/bge-reranker-v2-m3` (~2.3 GB) | 3–5 минут впервые |

### Anchor-based context selection (post-processing после reranker)

Reranker отлично решает задачу «вытащить нужный фрагмент в top-1»: на основном наборе у нас `document_hit@1 = 0.9538`, `document_mrr = 0.9744`. Но `document_precision@5` остаётся ~0.60, потому что после anchor'а в top-5 reranker всё равно тащит «клинически близкие, но неправильные» документы.

Решение — **anchor-based context selection**: top-1 reranker'a фиксируется как «якорный фрагмент», а оставшиеся 4 позиции в top-5 заполняются **связным контекстом вокруг anchor'а** — соседями по `chunk_index`, чанками того же раздела, страницы или документа. Это нормальная RAG-практика: LLM получает локально связный контекст вместо разрозненных результатов.

В научной формулировке для ВКР:

> После reranking наиболее релевантный фрагмент используется как якорный. Для повышения связности контекста и точности цитирования в финальный top-5 дополнительно включаются соседние фрагменты из того же раздела, страницы или документа. Это позволяет передавать LLM не разрозненные результаты поиска, а локально связанный клинический контекст.

#### Флаг `--context-selection`

| Mode | Что добирает в top-5 |
|---|---|
| `none` *(по умолчанию)* | ничего, top-K reranker'a без изменений |
| `anchor_section` | только чанки из того же документа **и** раздела, что у anchor'а |
| `anchor_page` | только чанки с пересекающимся диапазоном страниц (с допуском `--context-page-tolerance ±N`) |
| `anchor_document` | любые чанки того же документа (приоритет: соседи по chunk_index → same-page → same-section → остальные) |

Алгоритм всегда сохраняет anchor на rank=1, добор идёт в порядке: остальные reranked-чанки внутри scope → соседи по chunk_index → same-page → same-section → same-document → fallback на reranker top-N.

#### Эксперимент: что запустить

Полный анализ — четыре прогона с одинаковыми остальными параметрами:

```bash
for MODE in none anchor_section anchor_document anchor_page; do
  python -m src.evaluate_with_reranker \
    --queries-path data/retrieval_eval_queries_plus_hard_v1.jsonl \
    --embedding-model-key e5_large \
    --embedding-config configs/embedding_models.yaml \
    --embeddings-dir outputs/embeddings/e5_large \
    --reranker-key bge_reranker_v2_m3 \
    --reranker-config configs/rerankers.yaml \
    --candidate-top-k 30 --final-top-k 5 \
    --context-selection $MODE \
    --device cuda \
    --output-dir outputs_rerank/e5_large_${MODE} \
    --overwrite
done
```

Сводная таблица по 4 прогонам — на одном языке Pandas:

```python
import pandas as pd
rows = []
for mode in ['none', 'anchor_section', 'anchor_document', 'anchor_page']:
    rows.append(pd.read_csv(f'outputs_rerank/e5_large_{mode}/rerank_comparison_metrics.csv').iloc[0])
pd.DataFrame(rows).set_index('context_selection')[
    ['final_document_precision_at_5','final_document_recall_at_5','final_document_mrr',
     'final_chunk_hit_at_5','final_section_hit_at_5','final_page_hit_at_5']
]
```

#### Что сохранится в `--output-dir`

Файлы те же, что и раньше, но содержание расширено:
- `detailed_results.jsonl` — теперь содержит `dense_top_results`, `reranked_top_results` **и** `final_top_results` (после context selection); у каждого результата в `final_top_results` есть `context_source` (`anchor` / `reranker` / `neighbor` / `same_section` / `same_page` / `same_document` / `reranker_fallback`).
- `rerank_comparison_metrics.csv` — три колонки на каждую метрику: `dense_*`, `reranked_*`, `final_*`, плюс `delta_*` (full pipeline) и `delta_cs_*` (вклад только context selection).
- `rerank_report.md` — таблица `Dense → Reranked → Final` с двумя дельтами.

#### Важная честность

Anchor-based context selection не лжёт о retrieval-качестве, но **улучшает только если anchor правильный**. На наших данных Hit@1 после reranker = 0.9538 — почти всегда правильный, поэтому ожидание положительное. Если на каком-то срезе Hit@1 после reranker'а низкий, context selection усугубит ошибку — поэтому не применяйте mode'ы без выбора reranker'а.

В отчёте всегда показываются три набора метрик (`dense → reranked → final`) — чтобы было видно вклад каждого этапа. Так и пишите в ВКР: «вклад reranker'а» (`reranked − dense`) и «вклад context selection» (`final − reranked`).

### Дальше

После выбора лучшего пайплайна **(embedding_model, reranker, context_selection)** запускаете `pgvector_export.py` (раздел 6). В pgvector кладутся только chunks + embeddings; reranker и context selection применяются на этапе запроса в RAG-сервисе.

## 5.2. LLM generation (RAG vs no-RAG)

После того как retrieval-пайплайн зафиксирован (e5_large + bge-reranker-v2-m3 + anchor_page), к нему присоединяется LLM-слой: ответы с цитированием источников + baseline без RAG для сравнения.

> На этом этапе **не считаются** faithfulness и hallucination rate — только генерируются ответы и готовятся пары RAG vs no-RAG для следующего evaluator'а.

### Файлы

```
src/llm_client.py             # унифицированный клиент: openai / ollama / mock
src/prompt_templates.py       # SYSTEM_PROMPT_MEDICAL_RAG, RAG_*, NO_RAG_* шаблоны
src/rag_generation.py         # RetrievalEngine + generate_rag_answer/no_rag/parse_llm_json/validate_citations
src/generate_answers.py       # CLI: пакетная генерация на JSONL-кейсах (рекуррентный JSONL-вывод, --resume)
src/compare_rag_vs_no_rag.py  # CLI: merge двух JSONL по case_id + сводный CSV
configs/llm.yaml              # провайдер / модель / api_key_env (без ключей в коде)
```

### Конфиг LLM

`configs/llm.yaml`:
```yaml
llm:
  provider: "openai_compatible"   # или ollama / mock
  model_name: "gpt-4o-mini"
  api_key_env: "OPENAI_API_KEY"   # имя env-переменной с ключом
  base_url: null                  # для self-hosted vLLM/LM Studio — задайте
  temperature: 0.1
  max_tokens: 1500
  timeout_sec: 120
  request_json_mode: true         # OpenAI response_format=json_object
```

Активный provider выбирается ключом `llm.provider`, который ссылается на запись в `providers.*`. Поддерживаются:
- `openai_compatible` — gpt-* через `openai` SDK; работает и с self-hosted vLLM/LM Studio (укажите `base_url`).
- `ollama` — локальный сервер `http://localhost:11434/api/chat`.
- `mock` — без сети, возвращает шаблонный валидный JSON. Полезен для проверки pipeline без ключей.

API-ключ читается из переменной окружения, имя задаётся в `api_key_env` — в `configs/` ключи не попадают.

### Установка ключа

```bash
export OPENAI_API_KEY="sk-..."
```

В DataSphere — лучше использовать секреты проекта: `Project settings → Secrets`, потом в ноутбуке `os.environ['OPENAI_API_KEY'] = SecretManager().get('openai_api_key')`.

### 1. RAG-генерация (5 кейсов для отладки)

```bash
python -m src.generate_answers \
  --cases-path data/retrieval_eval_queries_plus_hard_v1.jsonl \
  --mode rag \
  --llm-config configs/llm.yaml \
  --embedding-model-key e5_large \
  --embedding-config configs/embedding_models.yaml \
  --embeddings-dir outputs/embeddings/e5_large \
  --reranker-key bge_reranker_v2_m3 \
  --reranker-config configs/rerankers.yaml \
  --candidate-top-k 30 --final-top-k 5 \
  --context-selection anchor_page \
  --device cuda \
  --output-path outputs_generation/rag_answers.jsonl \
  --limit 5 --overwrite
```

Что происходит:
1. Загружается `RetrievalEngine` (embedding + reranker).
2. На каждый case: dense top-30 → rerank → context_selection top-5 → формируется prompt с `[Источник S1]…[Источник S5]` → отправляется в LLM с `response_format=json_object`.
3. JSON парсится (поддерживает чистый JSON, markdown-блок, текст вокруг JSON).
4. Citations валидируются — каждая ссылка должна указывать на существующий `source_id` / `chunk_id` / `document_id` из retrieved.
5. Результат записывается **построчно** в JSONL (после каждого case'а — `flush`), так что прогресс не теряется при сбое. Один сбой одного case'а не роняет прогон.

### 2. No-RAG baseline

```bash
python -m src.generate_answers \
  --cases-path data/retrieval_eval_queries_plus_hard_v1.jsonl \
  --mode no_rag \
  --llm-config configs/llm.yaml \
  --output-path outputs_generation/no_rag_answers.jsonl \
  --limit 5 --overwrite
```

В no-RAG-режиме prompt просит модель выдать тот же JSON, но **без источников**: `citations` должны быть пустыми массивами. Валидатор citations проверяет, что модель не выдумала ссылки.

### 3. Объединение пар

```bash
python -m src.compare_rag_vs_no_rag \
  --rag-path outputs_generation/rag_answers.jsonl \
  --no-rag-path outputs_generation/no_rag_answers.jsonl \
  --output-pairs outputs_generation/rag_vs_no_rag_pairs.jsonl \
  --output-summary outputs_generation/rag_vs_no_rag_summary.csv
```

Появятся:
- `rag_vs_no_rag_pairs.jsonl` — на каждый `case_id` объект `{case_id, patient_case, rag, no_rag}`. Это полная разметка для следующего faithfulness-evaluator'а.
- `rag_vs_no_rag_summary.csv` — таблица для быстрой ручной проверки. Колонки: `rag_citation_count`, `rag_invalid_citation_count`, `rag_citation_coverage_estimate`, `rag_num_diagnoses`, `rag_insufficient_information`, `rag_retrieved_document_ids` и аналогичные для no_rag.

### Что лежит в одной записи `rag_answers.jsonl`

```json
{
  "case_id": "q001",
  "patient_case": "...",
  "mode": "rag",
  "context_selection": "anchor_page",
  "answer_json": { "differential_diagnoses": [...], ... },
  "answer_raw_text": "<сырой текст модели>",
  "retrieved_chunks": [
    {"source_id":"S1","chunk_id":"...","document_id":"...","section_title":"...",
     "page_start":11,"page_end":11,"text":"...","dense_score":...,
     "reranker_score":...,"context_source":"anchor"}
  ],
  "llm": {"provider":"openai_compatible","model_name":"gpt-4o-mini","usage":{...}},
  "citation_validation": {
    "valid": true,
    "citation_count": 6,
    "invalid_citation_count": 0,
    "citation_coverage_estimate": 1.0,
    "errors": []
  },
  "timing": {"retrieval_time_sec":0.12,"generation_time_sec":3.4,"total_time_sec":3.55},
  "errors": []
}
```

### Как проверить результат

1. **JSON-валидность.** `head -1 outputs_generation/rag_answers.jsonl | jq .answer_json.differential_diagnoses[0]` — должна быть структура с `diagnosis`, `probability_level`, `citations`.
2. **Citation validity.** `summary.csv` колонка `rag_invalid_citation_count` — должна быть 0 для большинства case'ов. Если ≥ 1 на каком-то case — модель выдумала источник; смотрите `citation_validation.errors` в детальном JSONL.
3. **Citation coverage.** `rag_citation_coverage_estimate` — доля diagnosis+recommendation items с хотя бы одной citation. На валидном RAG-прогоне ожидается ≥ 0.9 (модель ссылается почти везде). Сильное падение — сигнал, что prompt не усвоился.
4. **Сравнение со no-RAG.** В summary `no_rag_citation_count` должен быть 0 (no-RAG не выдумывает источники). `rag_num_diagnoses` vs `no_rag_num_diagnoses` — насколько RAG конкретизирует ответ.
5. **Покрытие документов retrieval'ом.** `rag_retrieved_document_ids` — в идеале должно пересекаться с `expected_document_ids` исходного запроса.

### Mock-режим для отладки без ключа

Меняете в `configs/llm.yaml`:
```yaml
llm:
  provider: "mock"
```
И запускаете тот же CLI. Mock-LLM возвращает шаблонный валидный JSON с citation на S1 (с реальным chunk_id из retrieved). Это удобно проверить pipeline до того, как тратить токены.

Можно временно переопределить через CLI без правки конфига:
```bash
python -m src.generate_answers ... --llm-provider mock
```

### Что НЕ входит в этот этап

- expert review;
- Telegram bot, REST API, frontend;
- запись в pgvector (об этом — раздел 6).

## 5.3. LLM evaluation (faithfulness / hallucination / citation accuracy / answer relevance)

Поверх готовых RAG/no-RAG ответов запускается LLM-as-a-judge оценка. Это этап 6
по ВКР: `пункты 5/6/7/8` пайплайна (передача в LLM, цитирование с
прослеживаемостью, тестовый набор клинических случаев, сравнение RAG vs no-RAG).

### Файлы

- `configs/prompts/judge_faithfulness_v1.yaml` — claim-level faithfulness/hallucination judge.
- `configs/prompts/judge_citation_accuracy_v1.yaml` — citation accuracy judge (RAG only).
- `configs/prompts/judge_answer_relevance_v1.yaml` — answer-level relevance/usefulness/safety judge.
- `configs/prompts/rag_diagnostic_v2_strict.yaml` — более строгий RAG prompt
  (citation на каждое существенное утверждение).
- `configs/prompts/no_rag_diagnostic_v2_baseline.yaml` — честный baseline без RAG
  (модель ОБЯЗАНА отвечать; citations=[]).
- `data/clinical_cases_v1.jsonl` — собранный набор клинических случаев (auto-converted из
  retrieval queries + manual hard cases). Кейсы с `review_status="auto_generated"`
  требуют ручной верификации формулировок и `expected_diagnoses`.

### Команда CLI

```bash
python -m src.evaluate_llm_answers \
  --pairs-path runs/exp_full_v1/generation/rag_vs_no_rag_pairs.jsonl \
  --clinical-cases-path data/clinical_cases_v1.jsonl \
  --llm-config configs/llm.yaml \
  --judge-prompt-config configs/prompts/judge_faithfulness_v1.yaml \
  --citation-judge-prompt-config configs/prompts/judge_citation_accuracy_v1.yaml \
  --relevance-judge-prompt-config configs/prompts/judge_answer_relevance_v1.yaml \
  --output-dir runs/exp_full_v1/llm_eval \
  --limit 30 --overwrite
```

Для смок-теста без сети: `--judge-provider mock --judge-model mock-llm`.

### Полный pipeline с LLM evaluation

```bash
python -m src.run_experiment \
  --config configs/experiments/exp_full_with_llm_eval_v1.yaml
```

Этот конфиг расширяет `exp_full_v1.yaml` и добавляет секцию `llm_evaluation:`
(вызывается после comparison-этапа).

### Что появится в `runs/<exp>/llm_eval/`

- `claim_evaluations.jsonl` — оценка каждого claim (одна строка на claim);
- `case_evaluations.jsonl` — per (case, mode) (faithfulness + citation + relevance);
- `rag_vs_no_rag_eval_summary.csv|.json` — построчное сравнение RAG vs no-RAG;
- `llm_eval_metrics.json|.csv` — агрегаты + comparison + targets;
- `llm_eval_report.md` — основной отчёт ВКР;
- `failed_judge_cases.jsonl` — кейсы, где judge упал/вернул невалидный JSON.

`llm_eval_metrics.json` также зеркалится в `runs/<exp>/metrics/llm_eval_metrics.json`.

### Target thresholds (ВКР)

| Цель | Порог |
|---|---|
| RAG faithfulness_soft | ≥ 0.85 |
| RAG hallucination_rate | < 0.15 |
| RAG citation_accuracy_rate | ≥ 0.90 |
| Faithfulness improvement (RAG − no-RAG) | ≥ 0.20 |
| Hallucination reduction (no-RAG − RAG) | ≥ 0.20 |

В `final_metrics_summary.md` есть таблица Pass/Fail (✓/✗) по этим целям.

### Подготовка clinical cases

Создание `data/clinical_cases_v1.jsonl` (одноразово):

```bash
python -m src.clinical_cases_tools \
  --input data/retrieval_eval_queries_plus_hard_v1.jsonl \
  --manual data/clinical_cases_manual.jsonl \
  --output data/clinical_cases_v1.jsonl \
  --review-csv data/clinical_cases_v1_review_template.csv \
  --mode convert
```

### Оговорка про LLM-as-a-judge

Это автоматическая оценка. Она НЕ заменяет ручную экспертно-врачебную проверку
ответов. Часть кейсов помечены `review_status="auto_generated"` и требуют
ручной разметки. Для пограничных или жизнеугрожающих сценариев финальная
валидация всегда должна делаться экспертом.

## 6. Подготовка pgvector export

После выбора лучшей модели — экспорт в формат для PostgreSQL + pgvector:

```bash
python -m src.pgvector_export \
  --model-key bge_m3 \
  --embeddings-dir outputs/embeddings/bge_m3 \
  --output-dir outputs/pgvector
```

Появятся:
- `outputs/pgvector/chunks_for_pgvector_bge_m3.jsonl` — каждая строка содержит и метаданные, и `embedding`, `embedding_model`, `embedding_dim`.
- `outputs/pgvector/schema_bge_m3.sql` — таблица `chunks` со столбцом `embedding vector(N)`, индексами по `document_id`/`label`/`specialty` и закомментированным IVFFlat / HNSW индексом.

---

## Notebooks

- `notebooks/01_validate_chunks.ipynb` — запуск и просмотр отчёта валидации.
- `notebooks/02_build_embeddings_one_model.ipynb` — шаблон для одного DataSphere-ноутбука: меняете `MODEL_KEY` в первой ячейке и запускаете.
- `notebooks/03_compare_embedding_models.ipynb` — финальный ноутбук: запускает `evaluate_models` и визуализирует таблицу.

---

## Идеальная последовательность работы

1. Подготовить общий архив (`src/`, `configs/`, `data/`, `requirements.txt`).
2. Загрузить архив в несколько Yandex DataSphere GPU-ноутбуков.
3. В каждом ноутбуке `pip install -r requirements.txt`.
4. В каждом ноутбуке запустить `build_embeddings.py` со своим `--model-key`.
5. Скачать/собрать папки `outputs/embeddings/*` в один общий проект.
6. В одном финальном ноутбуке запустить `evaluate_models.py`.
7. Прочитать `outputs/reports/embedding_model_comparison.csv` и `best_embedding_model.json`.
8. По победителю — `pgvector_export.py`.

---

## 9. REST API для RAG retrieval модуля (ВКР, пункт 9)

**Цель:** REST API поиска и выдачи релевантных фрагментов с временем ответа не более 2 секунд при базе до 5000 фрагментов.

### Структура API-модуля

```
configs/
  api.yaml                  # конфигурация API (модели, порты, лимиты)
src/api/
  __init__.py
  main.py                   # FastAPI app, lifespan, все endpoints
  schemas.py                # Pydantic v2 schemas
  retrieval_service.py      # RetrievalService (загрузка + поиск)
  health.py                 # health check helpers
  benchmark_api.py          # бенчмарк /search endpoint
  smoke_test_api.py         # дымовой тест всех endpoints
```

### 9.1. Установка зависимостей

```bash
cd rag_retrieval
pip install -r requirements.txt
```

Ключевые добавленные зависимости: `fastapi>=0.110`, `uvicorn[standard]>=0.27`, `pydantic>=2.0`.

### 9.2. Запуск API

```bash
# Способ 1 — через CLI (читает конфиг из configs/api.yaml)
python -m src.api.main --config configs/api.yaml

# Способ 2 — через uvicorn напрямую (конфиг через env)
RAG_API_CONFIG=configs/api.yaml uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# С кастомным хостом/портом
python -m src.api.main --config configs/api.yaml --host 0.0.0.0 --port 8080
```

При старте сервис:
1. Загружает embeddings (`.npy` + `metadata.jsonl`).
2. Загружает embedding model (`e5_large` на GPU/CPU).
3. Загружает reranker (`bge_reranker_v2_m3`).
4. Прогревает pipeline warmup-запросом.
5. Предвычисляет кеш документов и статистики.

### 9.3. Проверка состояния

```bash
curl http://localhost:8000/health
```

Ответ:
```json
{
  "status": "ok",
  "service": "rag-retrieval-api",
  "loaded": true,
  "embedding_model_key": "e5_large",
  "reranker_key": "bge_reranker_v2_m3",
  "num_chunks": 3702,
  "device": "cuda",
  "uptime_sec": 15.3
}
```

```bash
curl http://localhost:8000/ready
```

```json
{
  "ready": true,
  "checks": {
    "embeddings_loaded": true,
    "metadata_loaded": true,
    "embedding_model_loaded": true,
    "reranker_loaded": true
  }
}
```

### 9.4. Поиск релевантных фрагментов

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Пациент 58 лет, давящая боль за грудиной при нагрузке",
    "candidate_top_k": 30,
    "final_top_k": 5,
    "use_reranker": true,
    "context_selection": "anchor_page",
    "include_text": true
  }'
```

Ответ содержит:
- `results` — топ-K фрагментов с метаданными (`chunk_id`, `document_id`, `document_title`, `section_title`, `label`, `page_start`, `page_end`, `source`, `text`), оценками (`dense_score`, `reranker_score`, `final_score`) и `context_source` (`anchor`, `reranker`, `neighbor`, `same_page` и др.).
- `timing` — время каждого этапа pipeline в мс.
- `trace_id` — UUID запроса для трассировки.

**Параметры запроса:**

| Поле | По умолчанию | Описание |
|---|---|---|
| `query` | — | Текст запроса (обязательно) |
| `candidate_top_k` | 30 | Размер пула кандидатов dense retrieval |
| `final_top_k` | 5 | Размер итоговой выдачи |
| `use_reranker` | true | Включить BGE reranker |
| `context_selection` | `anchor_page` | `none` / `anchor_page` / `anchor_section` / `anchor_document` |
| `filters.document_id` | null | Фильтр по документу |
| `filters.label` | null | Фильтр по типу раздела |
| `filters.specialty` | null | Фильтр по специальности |
| `include_text` | true | Включить текст фрагментов в ответ |
| `include_embedding_text` | false | Включить embedding_text |

### 9.5. Пакетный поиск

```bash
curl -X POST http://localhost:8000/batch_search \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      "Пациент 58 лет, давящая боль за грудиной",
      "Лечение фибрилляции предсердий"
    ],
    "candidate_top_k": 30,
    "final_top_k": 5,
    "use_reranker": true,
    "context_selection": "anchor_page",
    "include_text": false
  }'
```

Максимальный размер батча: 20 запросов (настраивается в `api.yaml`).

### 9.6. Дополнительные endpoints

```bash
# Статистика базы
curl http://localhost:8000/stats

# Список документов
curl http://localhost:8000/documents

# Активная конфигурация
curl http://localhost:8000/config
```

### 9.7. Дымовой тест

```bash
python -m src.api.smoke_test_api \
  --url http://localhost:8000 \
  --query "Пациент 58 лет, боль за грудиной при нагрузке"
```

Проверяет:
- `/health` — статус ok, loaded=true, num_chunks > 0
- `/ready` — все компоненты загружены
- `/search` — возвращает ≥1 результата с корректными метаданными
- `/search` без reranker — быстрый режим
- `/stats` — num_chunks > 0
- `/documents` — список документов не пустой
- Время ответа `/search` < 2000 мс

### 9.8. Бенчмарк задержки

```bash
python -m src.api.benchmark_api \
  --url http://localhost:8000/search \
  --queries-path data/clinical_cases_v1.jsonl \
  --query-field patient_case \
  --limit 50 \
  --output runs/api_benchmark/api_benchmark_results.json
```

Опции:

| Флаг | По умолчанию | Описание |
|---|---|---|
| `--limit N` | все | Первые N запросов |
| `--concurrency N` | 1 | Параллельность (для ВКР достаточно 1) |
| `--no-reranker` | — | Пропустить reranker (быстрый режим) |
| `--target-latency-ms` | 2000 | Целевая задержка для отчёта |

**Результаты сохраняются в:**

```
runs/api_benchmark/
  api_benchmark_results.json   # детальные результаты + статистика
  api_benchmark_results.csv    # per-query: latency, success, num_results
  api_benchmark_report.md      # читаемый отчёт с таблицей метрик
```

**Метрики бенчмарка:**

| Метрика | Описание |
|---|---|
| `num_requests` | Всего запросов |
| `success_rate` | Доля успешных (HTTP 200) |
| `avg_latency_ms` | Среднее время ответа |
| `p50_latency_ms` | Медиана |
| `p95_latency_ms` | 95-й перцентиль |
| `max_latency_ms` | Максимум |
| `target_pass_rate` | Доля запросов < 2000 мс |

### 9.9. Соответствие требованиям ВКР

Данная реализация удовлетворяет пункту 9 ВКР:

| Требование | Реализация |
|---|---|
| REST API реализован | FastAPI + uvicorn, endpoints: `/health`, `/ready`, `/search`, `/batch_search`, `/stats`, `/documents`, `/config` |
| Поиск и выдача фрагментов | POST `/search` возвращает top-K chunks с ранжированием |
| Метаданные с полями цитирования | `chunk_id`, `document_id`, `document_title`, `section_title`, `label`, `page_start`, `page_end`, `source` в каждом результате |
| Прослеживаемость контекста | `context_source` показывает, как фрагмент попал в выдачу (`anchor`, `reranker`, `neighbor`, `same_page`, ...) |
| Время ответа < 2 сек | Подтверждается бенчмарком (dense + rerank + context_selection на 3700 чанков) |
| База до 5000 фрагментов | Архитектура numpy-based retrieval масштабируется до ≥5000 без потери скорости |
| Warmup | Сервис прогревается при старте; первый боевой запрос уже работает быстро |
