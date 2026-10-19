# Simulating Human Memory with Language Models

This repository contains data and code for the paper, [Simulating Human Memory with Language Models](https://arxiv.org/abs/2605.25680)

> Abstract: Language models are increasingly being deployed as user simulators, but their memory is far more reliable than that of real users. To measure this gap, we run a series of classic memory experiments from psychology on both humans and language models. Across tasks, we find that out-of-the-box language models exhibit better memory than humans, even when prompted to imitate human behavior. We then show that better prompting strategies and the use of a compactor can cause language models to forget content in a more human-like way.  Using these methods, we show preliminary evidence that language models with human-like memory constraints can function as more effective user simulators in a downstream education task. Finally, we release human reference data and benchmarks to support future work on simulating human memory with language models.

The benchmark includes ten tasks covering working memory, recall, recognition,
and reasoning over remembered information. You can score the bundled model
outputs against human reference data or run the tasks on another model.
Prompting conditions ask models to complete the task directly or simulate a human
participant. The **Compactor** condition gives the model a four-slot key-value
memory store for encoding information before answering questions.

## Install

Use Python 3.10 or later from a clone of this repository. An editable installation
keeps the code connected to the bundled stimuli and run files.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The default installation supports **offline scoring** and only requires NumPy.
To run new model experiments, install the additional dependencies:

```bash
python -m pip install -e ".[experiments]"
```

## Score the released data

No API key or model requests are needed. From the repository root:

```bash
python src/score.py
# Equivalent after installation: memory-score
```

This scores the saved prompting and Compactor runs against the default human
reference: **100 completed participants per task**, with 99 usable free-recall
scores. It writes
`results/scores.txt`, `results/scores.csv`, and `results/scores.json`.
Each row includes the human and model sample counts, normalized means, exact
Wasserstein distance, and humanlikeness (`1 - distance`). JSON also records the
scoring settings and any data limitations. Missing comparisons are labeled, and
missing values are written as JSON `null`.

Score a new model or an experiment folder containing several models:

```bash
python src/score.py --model-dir runs/my-experiment \
  --tasks digit_span_forward craft_task --conditions C1 C2 C3 compactor \
  --out results/my-experiment.txt
```

`--model-dir` can be repeated. It accepts a model directory, its `tasks/`
subdirectory, or a parent containing model directories. Task and condition lists
accept spaces or commas. Use `--include-samples` to include the underlying score
arrays in JSON, or `--human-root PATH` to supply another human reference dataset.

Two human reference datasets are available. The default, selected with
`--human-cohort expanded`, contains the 100-participant-per-task cohort. Use
`--human-cohort original` for the smaller reference dataset used in the paper:

```bash
python src/score.py --human-cohort original --out results/original-cohort.txt
```

The cohort option also selects the session policy described below.
`--human-selection latest|paper` overrides that policy independently of the
dataset.

## Scoring definitions

Scoring functions are implemented in `bench/core/scoring.py`. Each task produces
a distribution of human scores and a distribution of model scores. Scores are
divided by the task's normalization scale before computing their empirical
1-Wasserstein distance. Humanlikeness is `1 - distance`; larger values indicate
closer agreement between the two distributions.

| Task | Observation used for comparison | Normalization scale |
|---|---|---:|
| Forward / reverse digit span | One best span per participant: at least one correct attempt at a length; stop when both attempts fail at the same length | 20 |
| N-back | One session: human summary accuracy; model mean accuracy over answered trials across the three n-level blocks | 1 |
| Word recognition | Trials reached through the third error, **including error trials** | 100 by default |
| Variable mapping | Highest relations-tracked score across attempts per participant | 10 |
| Factual / narrative QA | Correct answers per participant | 10 |
| Narrative free recall | Saved embedding similarity per recall | 1 |
| Map / craft | Sum of correct answers over the three trials per participant | 15 |

The default `latest` session policy uses the latest completed session per
participant for digit span, n-back, word recognition, and free recall. For
variable mapping, QA, map, and craft, it uses the highest score across completed
runs per participant. The `paper` policy retains all completed sessions for digit
span, n-back, word recognition, and free recall, and uses the same highest-score
rule for the remaining tasks. Incomplete sessions are excluded in both cases.

Details that affect interpretation:

- Bundled Compactor digit-span files store many sequences under a single
  participant ID. Scoring uses the first two sequences per length and reports
  **one participant score**, with a diagnostic. Runs launched with `src/run.py`
  record distinct participants with two attempts per length. A span ends when
  both attempts fail; later generated trials do not extend the best span.
- Word-recognition runs have different trial caps. The default normalization
  scale is 100; `--word-recognition-scale 200` selects a scale of 200 for both
  humans and models. Scores exceeding the chosen scale can produce normalized
  values above 1. Scores and humanlikeness are not clipped.
- Two of the 100 selected human craft sessions use crafting systems that differ
  from those in `data/craft_task.json`. Use `--craft-version current` to restrict
  the reference to the matching systems (98 human participants). The smaller
  `original` cohort contains five sessions using different systems. Craft scores
  combine all three trials into a total out of 15.
- The default free-recall reference excludes one missing embedding score,
  leaving **99 scored participants**. Missing model embedding similarities are
  scored as zero and reported in the scoring notes.
- Compactor n-back and variable-mapping runners retain conversation history.
  Word recognition includes the full word list at recall, including in its
  summarizer variant. These tasks expose information beyond the memory store,
  so their results do not measure performance under an isolated four-slot
  memory constraint.

## Run a new experiment

Set the API credential expected by your backend (`OPENAI_API_KEY`,
`OPENROUTER_API_KEY` for an OpenRouter base URL, or `ANTHROPIC_API_KEY` for the
Anthropic backend). Substitute an available model ID for `YOUR_MODEL_ID`.
Preview the configuration without loading a model or making requests:

```bash
python src/run.py --model YOUR_MODEL_ID --out-dir runs/my-experiment \
  --tasks digit_span_forward,craft_task --repeat 5 --dry-run
```

Run those tasks and score the saved results afterwards:

```bash
python src/run.py --model YOUR_MODEL_ID --out-dir runs/my-experiment \
  --tasks digit_span_forward,craft_task --repeat 5 --score
```

Useful controls:

- `--repeat N`: simulated participants (default 5). For Compactor free recall,
  this is repeats per story.
- `--include-compactor`: add the Compactor variants of the selected tasks.
  `--compactor-only` runs just those variants. Explicit `wm_` task names also work.
- `--max-parallel-participants N` and `--max-parallel-tasks N`: control concurrency.
- `--backend anthropic`: direct Anthropic access; otherwise use the
  OpenAI-compatible backend. `--base-url https://openrouter.ai/api/v1` selects
  OpenRouter. `--qwen-thinking true|false` configures supported Qwen3-8B runs.
- `--score`: write score reports in the completed model directory using the
  default human reference.
- Existing task output directories require a new `--out-dir` or explicit
  `--overwrite`; the underlying runner replaces task files rather than resuming.

Prompting task runners execute four conditions:

| Condition | Description | Scorer label |
|---|---|---|
| C1 | Describe the task | TaskPr |
| C2 | Ask the model to simulate a human participant | HumPr |
| C3 | Simulate a human with limited memory | MemPr |
| C4 | Include examples of human responses | Not included in score reports |

Compactor outputs use C2 internally and are reported as `compactor`. Outputs are
stored under `<out-dir>/<model-slug>/tasks/` as JSONL, with a configuration
snapshot alongside them. The scorer reads these trial records to compute the
score distributions.

## Use scoring in Python

```python
from pathlib import Path
from bench.core.scoring import human_scores, llm_scores, humanlikeness

humans = human_scores("craft_task", craft_version="current")
model = llm_scores("craft_task", Path("runs/my-experiment/my-model"), "C1")
print(len(humans), len(model), humanlikeness(humans, model))
```

Functions return normalized NumPy arrays. A model directory needs a `tasks/`
folder with `<task>.jsonl` for C1–C3 and `wm_<task>.jsonl` for Compactor. Each row
must retain the benchmark's participant/session IDs and trial-level scores;
changing those identifiers changes the distribution being evaluated.

## Released data and layout

- `bench/`: task runners, prompts, and reusable scoring functions.
- `src/`: `run.py` and `score.py` command-line tools.
- `data/`: experimental stimuli.
- `runs/human/`: the smaller human reference dataset used in the paper.
- `runs/human-2026-07-27/`: 1,225 task records, including repeated/incomplete
  sessions; 100 completed participants per task and 110 distinct participants
  across tasks. This is the default reference, with usable free-recall n=99.
- `runs/prompting/` and `runs/compactor/`: saved model outputs, including four-slot
  Gemini 3.1 Pro Preview and Gemini 3.6 Flash Compactor runs. Gemini prompt-only
  runs are not included.
- `runs/data_update_2026-07-27.json`: source commit, cohort notes, and SHA-256
  checksums for every imported file.
- `application/`: the independent educational document-ranking experiment. Its
  setup and usage are described in [application/README.md](application/README.md).

Human participant IDs are pseudonyms that can be joined across task folders.
Human records use pseudonymous filenames and exclude network/browser details,
completion codes, feedback, and absolute collection timestamps. Completion order
is retained for selecting sessions. Participant pseudonyms in
`runs/human-2026-07-27/` are consistent with matching records in `runs/human/`.
The application study uses its own participant pseudonyms.

## Checks

```bash
python -m pip install -e ".[experiments,test]"
python -m unittest discover -s tests -v
```

Tests cover stopping and grouping rules, exact Wasserstein against SciPy, CLI
routing with mocked backends, cohort counts, and imported-file integrity. They do
not call model APIs. Runner tests require the optional experiment dependencies;
score extraction itself does not.

## License

Code is released under the MIT License. Benchmark data is released under CC BY 4.0.

## Citation
```
@misc{wang2026simulatinghumanmemorylanguage,
      title={Simulating Human Memory with Language Models}, 
      author={Qihan Wang and Nicholas Tomlin and Michael Hu and Brian Dillon and Tal Linzen},
      year={2026},
      eprint={2605.25680},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.25680}, 
}
```
