# Yandex DataSphere Pipeline Commands

> Шаблон. Конкретные команды для эксперимента генерируются автоматически:
>
> ```bash
> python -m src.write_datasphere_commands \
>   --config configs/experiments/<experiment>.yaml \
>   --output runs/<experiment>/datasphere_pipeline_commands.md
> ```
>
> Также этот файл создаётся автоматически в `runs/<experiment>/datasphere_pipeline_commands.md`
> при запуске `run_experiment.py`.

## 0. Проверка окружения

```bash
python --version
nvidia-smi
pip install -r requirements.txt
```

## 1. Полный эксперимент (одной командой)

```bash
python -m src.run_experiment \
  --config configs/experiments/{{EXPERIMENT_NAME}}.yaml
```

После запуска все артефакты — в `{{OUTPUT_DIR}}/`.

## 2. Этапы по отдельности

### 2.1. Retrieval evaluation (несколько embedding-моделей)

```bash
python -m src.evaluate_models \
  --queries-path {{RETRIEVAL_QUERIES_PATH}} \
  --config {{EMBEDDING_CONFIG}} \
  --embeddings-root {{EMBEDDINGS_ROOT}} \
  --model-keys {{MODEL_KEYS}} \
  --output-dir {{OUTPUT_DIR}}/retrieval \
  --top-k {{TOP_K}}
```

### 2.2. Reranking + context selection

```bash
python -m src.evaluate_with_reranker \
  --queries-path {{RETRIEVAL_QUERIES_PATH}} \
  --embedding-model-key {{EMBEDDING_MODEL_KEY}} \
  --embedding-config {{EMBEDDING_CONFIG}} \
  --embeddings-dir {{EMBEDDINGS_DIR}} \
  --reranker-key {{RERANKER_KEY}} \
  --reranker-config {{RERANKER_CONFIG}} \
  --candidate-top-k {{CANDIDATE_TOP_K}} --final-top-k {{FINAL_TOP_K}} \
  --context-selection {{CONTEXT_SELECTION}} \
  --device {{DEVICE}} \
  --output-dir {{OUTPUT_DIR}}/rerank \
  --overwrite
```

### 2.3. RAG generation

```bash
python -m src.generate_answers \
  --cases-path {{CLINICAL_CASES_PATH}} \
  --case-id-field {{CASE_ID_FIELD}} \
  --patient-case-field {{PATIENT_CASE_FIELD}} \
  --mode rag \
  --llm-config {{LLM_CONFIG}} \
  --prompt-config {{PROMPT_CONFIG_RAG}} \
  --embedding-model-key {{EMBEDDING_MODEL_KEY}} \
  --embedding-config {{EMBEDDING_CONFIG}} \
  --embeddings-dir {{EMBEDDINGS_DIR}} \
  --reranker-key {{RERANKER_KEY}} \
  --reranker-config {{RERANKER_CONFIG}} \
  --candidate-top-k {{CANDIDATE_TOP_K}} --final-top-k {{FINAL_TOP_K}} \
  --context-selection {{CONTEXT_SELECTION}} \
  --device {{DEVICE}} \
  --output-path {{OUTPUT_DIR}}/generation/rag_answers.jsonl \
  --limit {{LIMIT}} --overwrite
```

### 2.4. No-RAG generation

```bash
python -m src.generate_answers \
  --cases-path {{CLINICAL_CASES_PATH}} \
  --case-id-field {{CASE_ID_FIELD}} \
  --patient-case-field {{PATIENT_CASE_FIELD}} \
  --mode no_rag \
  --llm-config {{LLM_CONFIG}} \
  --prompt-config {{PROMPT_CONFIG_NO_RAG}} \
  --output-path {{OUTPUT_DIR}}/generation/no_rag_answers.jsonl \
  --limit {{LIMIT}} --overwrite
```

### 2.5. RAG vs no-RAG comparison

```bash
python -m src.compare_rag_vs_no_rag \
  --rag-path {{OUTPUT_DIR}}/generation/rag_answers.jsonl \
  --no-rag-path {{OUTPUT_DIR}}/generation/no_rag_answers.jsonl \
  --output-pairs {{OUTPUT_DIR}}/generation/rag_vs_no_rag_pairs.jsonl \
  --output-summary {{OUTPUT_DIR}}/generation/rag_vs_no_rag_summary.csv
```

### 2.6. RAG vs no-RAG comparison

(см. секцию 2.5)

### 2.7. LLM-as-a-judge evaluation (faithfulness + citation accuracy + relevance)

```bash
python -m src.evaluate_llm_answers \
  --pairs-path {{OUTPUT_DIR}}/generation/rag_vs_no_rag_pairs.jsonl \
  --clinical-cases-path {{CLINICAL_CASES_PATH}} \
  --llm-config {{LLM_CONFIG}} \
  --judge-prompt-config {{JUDGE_PROMPT_CONFIG}} \
  --citation-judge-prompt-config {{CITATION_JUDGE_PROMPT_CONFIG}} \
  --relevance-judge-prompt-config {{RELEVANCE_JUDGE_PROMPT_CONFIG}} \
  --output-dir {{OUTPUT_DIR}}/llm_eval \
  --limit {{LLM_EVAL_LIMIT}} \
  --mode {{LLM_EVAL_MODE}} \
  --max-claims-per-answer {{LLM_EVAL_MAX_CLAIMS}} \
  --overwrite
```

Для smoke без сети используйте `--judge-provider mock --judge-model mock-llm`.

### 2.8. Сборка summary

```bash
python -m src.collect_experiment_summary \
  --output-dir {{OUTPUT_DIR}}
```

## 3. Что смотреть после прогона

- `{{OUTPUT_DIR}}/metrics/pipeline_status.json` — статус каждого этапа.
- `{{OUTPUT_DIR}}/metrics/final_metrics_summary.md` — итоговые метрики (включая LLM evaluation + Target thresholds).
- `{{OUTPUT_DIR}}/llm_eval/llm_eval_report.md` — детальный LLM-evaluation отчёт.
- `{{OUTPUT_DIR}}/llm_eval/rag_vs_no_rag_eval_summary.csv` — per-case Δ метрики.
- `{{OUTPUT_DIR}}/summary/experiment_summary.md` — отчёт человеку.
- `{{OUTPUT_DIR}}/rerank/rerank_report.md` — reranker-отчёт.
- `{{OUTPUT_DIR}}/generation/rag_vs_no_rag_summary.csv` — сравнение по case'ам (technical).
