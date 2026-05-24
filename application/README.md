# Application Reading QA

This folder contains an independent reading-comprehension evaluator for files in `application/documents`.

## What it does

- Runs condition-style prompts similar to `factual_qa`: `C1`, `C2`, `C3`, `C4`
- Randomly samples one document and one level per repeat:
  - `biography` -> `biography_text`
  - `reading_level` -> `reading_level.updated_biography`
  - `redundant` -> `redundancy.updated_biography`
  - `distractor` -> `distractor.updated_biography`
- Sends the selected reading plus all multiple-choice questions to the LLM.
- Appends one extra prompt item:
  - `On a scale 1-10, how difficult is the reading?`
- Computes:
  - QA accuracy (`correct / total`)
  - difficulty statistics (mean/std/min/max + 1-10 histogram)
  - grouped summaries by condition and by level

## Run

Activate virtual env first:

```bash
source .venv/bin/activate
```

Basic run:

```bash
python -m application.reading_qa.cli run --model openai/gpt-4.1-mini --n-repeat 20
```

Parallel requests (recommended for speed):

```bash
python -m application.reading_qa.cli run \
  --model openai/gpt-4.1-mini \
  --n-repeat 100 \
  --max-parallel-repeats 20
```

Custom output directory:

```bash
python -m application.reading_qa.cli run \
  --model openai/gpt-4.1-mini \
  --n-repeat 20 \
  --out-dir application/out/my_run
```

Optional backend/base URL override:

```bash
python -m application.reading_qa.cli run \
  --backend openai \
  --base-url https://openrouter.ai/api/v1 \
  --model qwen/qwen3-8b \
  --n-repeat 20
```

Qwen3-8B reasoning toggle (OpenRouter/OpenAI-compatible endpoints):

```bash
# thinking enabled
python -m application.reading_qa.cli run \
  --backend openai \
  --base-url https://openrouter.ai/api/v1 \
  --model qwen/qwen3-8b \
  --qwen-thinking true \
  --n-repeat 20

# thinking disabled
python -m application.reading_qa.cli run \
  --backend openai \
  --base-url https://openrouter.ai/api/v1 \
  --model qwen/qwen3-8b \
  --qwen-thinking false \
  --n-repeat 20
```

When `--qwen-thinking` is set for `qwen/qwen3-8b`, outputs are separated automatically:
- `application/out/qwen_qwen3-8b_thinking_true/`
- `application/out/qwen_qwen3-8b_thinking_false/`

## Output

By default output goes to `application/out/<model_slug>/tasks/`:

- `application_reading_qa.jsonl`: one row per repeat
- `application_reading_qa_summary.json`: overall + by-level metrics

## Plot from summary JSON

```bash
python -m application.reading_qa.cli plot \
  --summary-json application/out/openai_gpt-4.1-mini/tasks/application_reading_qa_summary.json
```

This writes PNG figures to:

- `application/out/openai_gpt-4.1-mini/tasks/figures/`
