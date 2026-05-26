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
  --config configs/experiments/exp_full_with_llm_eval_v1.yaml
```

После запуска все артефакты — в `runs/exp_full_with_llm_eval_v1/`.

## 2. Этапы по отдельности

### 2.1. Retrieval evaluation (несколько embedding-моделей)

```bash
python -m src.evaluate_models \
  --queries-path data/retrieval_eval_queries_plus_hard_v1.jsonl \
  --config configs/embedding_models.yaml \
  --embeddings-root outputs/embeddings \
  --model-keys e5_large e5_large_instruct bge_m3 mpnet_multilingual \
  --output-dir runs/exp_full_with_llm_eval_v1/retrieval \
  --top-k 10
```

### 2.2. Reranking + context selection

```bash
python -m src.evaluate_with_reranker \
  --queries-path data/retrieval_eval_queries_plus_hard_v1.jsonl \
  --embedding-model-key e5_large \
  --embedding-config configs/embedding_models.yaml \
  --embeddings-dir outputs/embeddings/e5_large \
  --reranker-key bge_reranker_v2_m3 \
  --reranker-config configs/rerankers.yaml \
  --candidate-top-k 30 --final-top-k 5 \
  --context-selection anchor_page \
  --device auto \
  --output-dir runs/exp_full_with_llm_eval_v1/rerank \
  --overwrite
```

### 2.3. RAG generation

```bash
python -m src.generate_answers \
  --cases-path data/clinical_cases_v1.jsonl \
  --case-id-field case_id \
  --patient-case-field patient_case \
  --mode rag \
  --llm-config configs/llm.yaml \
  --prompt-config configs/prompts/rag_diagnostic_v2_strict.yaml \
  --embedding-model-key e5_large \
  --embedding-config configs/embedding_models.yaml \
  --embeddings-dir outputs/embeddings/e5_large \
  --reranker-key bge_reranker_v2_m3 \
  --reranker-config configs/rerankers.yaml \
  --candidate-top-k 30 --final-top-k 5 \
  --context-selection anchor_page \
  --device auto \
  --output-path runs/exp_full_with_llm_eval_v1/generation/rag_answers.jsonl \
  --limit 20 --overwrite
```

### 2.4. No-RAG generation

```bash
python -m src.generate_answers \
  --cases-path data/clinical_cases_v1.jsonl \
  --case-id-field case_id \
  --patient-case-field patient_case \
  --mode no_rag \
  --llm-config configs/llm.yaml \
  --prompt-config configs/prompts/no_rag_diagnostic_v2_baseline.yaml \
  --output-path runs/exp_full_with_llm_eval_v1/generation/no_rag_answers.jsonl \
  --limit 20 --overwrite
```

### 2.5. RAG vs no-RAG comparison

```bash
python -m src.compare_rag_vs_no_rag \
  --rag-path runs/exp_full_with_llm_eval_v1/generation/rag_answers.jsonl \
  --no-rag-path runs/exp_full_with_llm_eval_v1/generation/no_rag_answers.jsonl \
  --output-pairs runs/exp_full_with_llm_eval_v1/generation/rag_vs_no_rag_pairs.jsonl \
  --output-summary runs/exp_full_with_llm_eval_v1/generation/rag_vs_no_rag_summary.csv
```

### 2.6. RAG vs no-RAG comparison

(см. секцию 2.5)

### 2.7. LLM-as-a-judge evaluation (faithfulness + citation accuracy + relevance)

```bash
python -m src.evaluate_llm_answers \
  --pairs-path runs/exp_full_with_llm_eval_v1/generation/rag_vs_no_rag_pairs.jsonl \
  --clinical-cases-path data/clinical_cases_v1.jsonl \
  --llm-config configs/llm.yaml \
  --judge-prompt-config configs/prompts/judge_faithfulness_v1.yaml \
  --citation-judge-prompt-config configs/prompts/judge_citation_accuracy_v1.yaml \
  --relevance-judge-prompt-config configs/prompts/judge_answer_relevance_v1.yaml \
  --output-dir runs/exp_full_with_llm_eval_v1/llm_eval \
  --limit 20 \
  --mode both \
  --max-claims-per-answer 20 \
  --overwrite
```

Для smoke без сети используйте `--judge-provider mock --judge-model mock-llm`.

### 2.8. Сборка summary

```bash
python -m src.collect_experiment_summary \
  --output-dir runs/exp_full_with_llm_eval_v1
```

## 3. Что смотреть после прогона

- `runs/exp_full_with_llm_eval_v1/metrics/pipeline_status.json` — статус каждого этапа.
- `runs/exp_full_with_llm_eval_v1/metrics/final_metrics_summary.md` — итоговые метрики (включая LLM evaluation + Target thresholds).
- `runs/exp_full_with_llm_eval_v1/llm_eval/llm_eval_report.md` — детальный LLM-evaluation отчёт.
- `runs/exp_full_with_llm_eval_v1/llm_eval/rag_vs_no_rag_eval_summary.csv` — per-case Δ метрики.
- `runs/exp_full_with_llm_eval_v1/summary/experiment_summary.md` — отчёт человеку.
- `runs/exp_full_with_llm_eval_v1/rerank/rerank_report.md` — reranker-отчёт.
- `runs/exp_full_with_llm_eval_v1/generation/rag_vs_no_rag_summary.csv` — сравнение по case'ам (technical).
