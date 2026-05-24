from __future__ import annotations
import json
import os
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import typer
import yaml

from .core.config import load_yaml
from .core.io import ensure_dir, write_json
from .core.llm_openai import OpenAIChatLLM
from .core.runner import run_tasks

from .tasks.digit_span_forward import evaluate as eval_forward
from .tasks.digit_span_reverse import evaluate as eval_reverse
from .tasks.nback import evaluate as eval_nback
from .tasks.semantic_story_recall import evaluate as eval_story
from .tasks.word_recognition import evaluate as eval_word_recognition
from .tasks.variable_mapping import evaluate as eval_variable_mapping
from .tasks.factual_qa import evaluate as eval_factual_qa
from .tasks.narrative_qa import evaluate as eval_narrative_qa
from .tasks.map_task import evaluate as eval_map_task
from .tasks.craft_task import evaluate as eval_craft_task
from .tasks.wm_semantic_story_recall import evaluate as eval_wm_story
from .tasks.wm_semantic_story_recall import evaluate_summarizer as eval_sum_story
from .tasks.wm_digit_span_forward import evaluate as eval_wm_digit_forward
from .tasks.wm_digit_span_forward import evaluate_summarizer as eval_sum_digit_forward
from .tasks.wm_digit_span_reverse import evaluate as eval_wm_digit_reverse
from .tasks.wm_digit_span_reverse import evaluate_summarizer as eval_sum_digit_reverse
from .tasks.wm_nback import evaluate as eval_wm_nback
from .tasks.wm_nback import evaluate_summarizer as eval_sum_nback
from .tasks.wm_word_recognition import evaluate as eval_wm_word_recognition
from .tasks.wm_word_recognition import evaluate_summarizer as eval_sum_word_recognition
from .tasks.wm_variable_mapping import evaluate as eval_wm_variable_mapping
from .tasks.wm_variable_mapping import evaluate_summarizer as eval_sum_variable_mapping
from .tasks.wm_factual_qa import evaluate as eval_wm_factual_qa
from .tasks.wm_factual_qa import evaluate_summarizer as eval_sum_factual_qa
from .tasks.wm_narrative_qa import evaluate as eval_wm_narrative_qa
from .tasks.wm_narrative_qa import evaluate_summarizer as eval_sum_narrative_qa
from .tasks.wm_application_reading_qa import evaluate as eval_wm_application_reading_qa
from .tasks.wm_map_task import evaluate as eval_wm_map_task
from .tasks.wm_map_task import evaluate_summarizer as eval_sum_map_task
from .tasks.wm_craft_task import evaluate as eval_wm_craft_task
from .tasks.wm_craft_task import evaluate_summarizer as eval_sum_craft_task

app = typer.Typer(add_completion=False)

TASKS = {
  "digit_span_forward": eval_forward,
  "digit_span_reverse": eval_reverse,
  "nback": eval_nback,
  "semantic_story_recall": eval_story,
  "word_recognition": eval_word_recognition,
  "variable_mapping": eval_variable_mapping,
  "factual_qa": eval_factual_qa,
  "narrative_qa": eval_narrative_qa,
  "map_task": eval_map_task,
  "craft_task": eval_craft_task,
  "wm_semantic_story_recall": eval_wm_story,
  "wm_digit_span_forward": eval_wm_digit_forward,
  "wm_digit_span_reverse": eval_wm_digit_reverse,
  "wm_nback": eval_wm_nback,
  "wm_word_recognition": eval_wm_word_recognition,
  "wm_variable_mapping": eval_wm_variable_mapping,
  "wm_factual_qa": eval_wm_factual_qa,
  "wm_narrative_qa": eval_wm_narrative_qa,
  "wm_application_reading_qa": eval_wm_application_reading_qa,
  "wm_map_task": eval_wm_map_task,
  "wm_craft_task": eval_wm_craft_task,
  "sum_semantic_story_recall": eval_sum_story,
  "sum_digit_span_forward": eval_sum_digit_forward,
  "sum_digit_span_reverse": eval_sum_digit_reverse,
  "sum_nback": eval_sum_nback,
  "sum_word_recognition": eval_sum_word_recognition,
  "sum_variable_mapping": eval_sum_variable_mapping,
  "sum_factual_qa": eval_sum_factual_qa,
  "sum_narrative_qa": eval_sum_narrative_qa,
  "sum_map_task": eval_sum_map_task,
  "sum_craft_task": eval_sum_craft_task,
}


def expand_out_dir_template(out_dir: str) -> str:
    """Expand date placeholders in configured output directories."""
    run_date = os.getenv("RUN_DATE", "").strip() or date.today().isoformat()
    return out_dir.replace("{date}", run_date).replace("{run_date}", run_date)


def _apply_model_extra_body_cli(
    model_cfg: dict[str, Any],
    *,
    extra_body_json: Optional[str],
    qwen_thinking: Optional[str],
) -> None:
    """Merge YAML ``model.extra_body`` with CLI ``--extra-body-json`` and optional ``--qwen-thinking``."""
    eb: dict[str, Any] = {}
    raw = model_cfg.get("extra_body")
    if isinstance(raw, dict):
        eb.update(raw)
    if extra_body_json:
        try:
            parsed = json.loads(extra_body_json.strip())
        except json.JSONDecodeError as e:
            raise typer.BadParameter(f"Invalid --extra-body-json: {e}") from e
        if not isinstance(parsed, dict):
            raise typer.BadParameter("--extra-body-json must be a JSON object (dict).")
        eb.update(parsed)
    if qwen_thinking is not None:
        name_l = str(model_cfg.get("name", "")).lower()
        if "qwen3-8b" in name_l:
            v = qwen_thinking.strip().lower()
            if v in ("true", "1", "yes"):
                eb["enable_thinking"] = True
            elif v in ("false", "0", "no"):
                eb["enable_thinking"] = False
            else:
                raise typer.BadParameter("--qwen-thinking must be true or false")
        else:
            typer.secho(
                f"Warning: --qwen-thinking ignored (model is {model_cfg.get('name')!r}; "
                'only applies when model id contains "qwen3-8b").',
                err=True,
            )
    if eb:
        model_cfg["extra_body"] = eb
    else:
        model_cfg.pop("extra_body", None)


@app.command()
def list_tasks():
    """List available tasks."""
    for k in TASKS.keys():
        typer.echo(k)

@app.command()
def run(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config."),
    task: List[str] = typer.Option([], "--task", "-t", help="Task name (repeatable). Empty = all."),
    model: str = typer.Option(None, "--model", help="Override model name (e.g., gpt-4o, gpt-4o-mini)."),
    out_dir: str = typer.Option(None, "--out-dir", help="Override run.out_dir from the config."),
    repeat: int = typer.Option(
        None,
        "--repeat",
        min=1,
        help="Override n_participants / n_repeat where the task uses simulated participants (see task_config).",
    ),
    debug: bool = typer.Option(False, "--debug", help="Print working memory state per event (wm_free_recall)."),
    story: str = typer.Option(None, "--story", help="Restrict wm_semantic_story_recall to one story (e.g. pieman, oregon_trail, baseball, eyespy)."),
    extra_body_json: Optional[str] = typer.Option(
        None,
        "--extra-body-json",
        help='Merge JSON object into model.extra_body (e.g. {"enable_thinking": false} for Qwen on OpenRouter).',
    ),
    qwen_thinking: Optional[str] = typer.Option(
        None,
        "--qwen-thinking",
        help='true or false: sets extra_body enable_thinking for Qwen3-8B (model id contains "qwen3-8b").',
    ),
):
    cfg = load_yaml(config)
    model_cfg = cfg.get("model", {})
    run_cfg = cfg.get("run", {})
    task_cfg = cfg.get("task_config", {})

    if model is not None:
        model_cfg["name"] = model
    if out_dir is not None:
        run_cfg["out_dir"] = out_dir
    if repeat is not None:
        run_cfg["repeat"] = int(repeat)

    _apply_model_extra_body_cli(
        model_cfg,
        extra_body_json=extra_body_json,
        qwen_thinking=qwen_thinking,
    )

    # reflect overrides in snapshot
    cfg["model"] = model_cfg
    cfg["run"] = run_cfg

    mpp = run_cfg.get("max_parallel_participants")

    # Derive a filesystem-safe model slug and nest output under it.
    model_name = model_cfg["name"]
    model_slug = model_name.replace("/", "_").replace("\\", "_")
    _eb_slug = model_cfg.get("extra_body")
    if isinstance(_eb_slug, dict) and isinstance(_eb_slug.get("enable_thinking"), bool):
        model_slug = f"{model_slug}_{'true' if _eb_slug['enable_thinking'] else 'false'}"
    out_dir_base = expand_out_dir_template(str(run_cfg.get("out_dir", "runs/latest")))
    run_cfg["out_dir"] = out_dir_base
    cfg["run"] = run_cfg
    out_dir = Path(out_dir_base) / model_slug
    ensure_dir(out_dir)

    write_json(out_dir/"config_snapshot.json", cfg)

    quiet = os.getenv("BENCH_QUIET", "").strip().lower() in ("1", "true", "yes")
    _eb = model_cfg.get("extra_body")
    _extra_body = _eb if isinstance(_eb, dict) and _eb else None
    backend = (model_cfg.get("backend") or "openai").lower()
    if backend == "anthropic":
        from .core.llm_anthropic import AnthropicChatLLM

        llm = AnthropicChatLLM(
            model=model_cfg["name"],
            api_key=model_cfg.get("api_key"),
            base_url=model_cfg.get("base_url"),
            show_request_progress=not quiet,
        )
    else:
        llm = OpenAIChatLLM(
            model=model_cfg["name"],
            api_key=model_cfg.get("api_key"),
            base_url=model_cfg.get("base_url"),
            extra_body=_extra_body,
            show_request_progress=not quiet,
        )

    chosen = task or list(TASKS.keys())
    fns=[]

    for tname in chosen:
        if tname not in TASKS:
            raise typer.BadParameter(f"Unknown task: {tname}. Use `bench list-tasks`.")
        fn = TASKS[tname]

        if tname=="free_recall":
            c = task_cfg.get("free_recall", {})
            # Mapping: repeat -> lists_per_length (override when provided)
            if repeat is not None:
                lists_per_length = int(repeat)
            else:
                lists_per_length = int(c.get("lists_per_length", 10))
            runner_fn = lambda fn=fn, c=c: fn(
                llm, out_dir,
                model_cfg=model_cfg,
                words_json_path=c.get("words_json_path","data/words.json"),
                list_lengths=c.get("list_lengths",[10,20,30]),
                lists_per_length=int(lists_per_length),
                stimuli_seed=int(c.get("stimuli_seed", run_cfg.get("seed",42))),
            )
            runner_fn.task_name = tname
            fns.append(runner_fn)
        elif tname=="digit_span_forward":
            c = task_cfg.get("digit_span_forward", {})
            # --repeat only sets n_participants (same as other tasks). sequences_per_span always from YAML.
            sequences_per_span = int(c.get("sequences_per_span", 2))
            if repeat is not None:
                n_participants_val = int(repeat)
            else:
                n_participants_val = int(c.get("n_participants", 1))
            runner_fn = lambda fn=fn, c=c, n_participants_val=n_participants_val, sequences_per_span=sequences_per_span, mpp=mpp: fn(
                llm, out_dir,
                model_cfg=model_cfg,
                stimuli_seed=int(c.get("stimuli_seed", run_cfg.get("seed",42))),
                min_span=int(c.get("min_span",2)),
                max_span=int(c.get("max_span",20)),
                sequences_per_span=int(sequences_per_span),
                n_participants=int(n_participants_val),
                max_parallel_participants=mpp,
            )
            runner_fn.task_name = tname
            fns.append(runner_fn)
        elif tname=="digit_span_reverse":
            c = task_cfg.get("digit_span_reverse", {})
            sequences_per_span = int(c.get("sequences_per_span", 2))
            if repeat is not None:
                n_participants_val = int(repeat)
            else:
                n_participants_val = int(c.get("n_participants", 1))
            runner_fn = lambda fn=fn, c=c, n_participants_val=n_participants_val, sequences_per_span=sequences_per_span, mpp=mpp: fn(
                llm, out_dir,
                model_cfg=model_cfg,
                stimuli_seed=int(c.get("stimuli_seed", run_cfg.get("seed",42))),
                min_span=int(c.get("min_span",2)),
                max_span=int(c.get("max_span",20)),
                sequences_per_span=int(sequences_per_span),
                n_participants=int(n_participants_val),
                max_parallel_participants=mpp,
            )
            runner_fn.task_name = tname
            fns.append(runner_fn)
        elif tname=="semantic_story_recall":
            c = task_cfg.get("semantic_story_recall", {})
            if repeat is not None:
                n_participants = int(repeat)
            else:
                n_participants = int(
                    c.get(
                        "n_participants",
                        c.get("n_repeat", c.get("repeats_per_stimulus", 1)),
                    )
                )
            stimuli_seed = int(c.get("stimuli_seed", run_cfg.get("seed", 42)))
            recall_max_tokens = c.get("recall_max_tokens")
            runner_fn = lambda fn=fn, c=c, n_participants=n_participants, stimuli_seed=stimuli_seed, mpp=mpp, recall_max_tokens=recall_max_tokens: fn(
                llm, out_dir,
                model_cfg=model_cfg,
                n_participants=n_participants,
                stimuli_seed=stimuli_seed,
                max_parallel_participants=mpp,
                recall_max_tokens=int(recall_max_tokens) if recall_max_tokens is not None else None,
            )
            runner_fn.task_name = tname
            fns.append(runner_fn)
        elif tname=="nback":
            c = task_cfg.get("nback", {})
            if repeat is not None:
                n_repeat_val = int(repeat)
            else:
                n_repeat_val = int(c.get("n_repeat", c.get("n_participants", 1)))
            runner_fn = lambda fn=fn, c=c, n_repeat_val=n_repeat_val, mpp=mpp: fn(
                llm, out_dir,
                model_cfg=model_cfg,
                n_repeat=n_repeat_val,
                n_repeats_per_participant=int(c.get("n_repeats_per_participant", 1)),
                stimuli_seed=int(c.get("stimuli_seed", 123)),
                temperature=float(c.get("temperature", 0)),
                max_parallel_participants=mpp,
            )
            runner_fn.task_name = tname
            fns.append(runner_fn)
        elif tname=="complex_span_math_letters":
            c = task_cfg.get("complex_span_math_letters", {})
            if repeat is not None:
                n_participants = int(repeat)
            else:
                n_participants = int(c.get("n_participants", 10))
            runner_fn = lambda fn=fn, c=c: fn(
                llm, out_dir,
                model_cfg=model_cfg,
                n_participants=n_participants,
                base_seed=int(c.get("base_seed", run_cfg.get("seed", 123))),
                temperature=float(c.get("temperature", model_cfg.get("temperature", 0))),
                calibration_trials=int(c.get("calibration_trials", 15)),
            )
            runner_fn.task_name = tname
            fns.append(runner_fn)
        elif tname == "word_recognition":
            c = task_cfg.get("word_recognition", {})
            if repeat is not None:
                n_repeat = int(repeat)
            else:
                n_repeat = int(c.get("n_repeat", c.get("n_participants", 1)))
            runner_fn = lambda fn=fn, c=c, mpp=mpp: fn(
                llm, out_dir,
                model_cfg=model_cfg,
                words_json_path=c.get("words_json_path", "data/words.json"),
                n_repeat=n_repeat,
                max_trials_per_game=int(c.get("max_trials_per_game", 100)),
                stimuli_seed=int(c.get("stimuli_seed", run_cfg.get("seed", 42))),
                max_parallel_participants=mpp,
            )
            runner_fn.task_name = tname
            fns.append(runner_fn)
        elif tname == "variable_mapping":
            c = task_cfg.get("variable_mapping", {})
            if repeat is not None:
                n_repeat = int(repeat)
            else:
                n_repeat = int(c.get("n_repeat", c.get("n_participants", 1)))
            runner_fn = lambda fn=fn, c=c, mpp=mpp: fn(
                llm, out_dir,
                model_cfg=model_cfg,
                names_json_path=c.get("names_json_path", "data/names.json"),
                city_json_path=c.get("city_json_path", "data/city.json"),
                n_repeat=n_repeat,
                n_runs_per_participant=int(c.get("n_runs_per_participant", 3)),
                stimuli_seed=int(c.get("stimuli_seed", run_cfg.get("seed", 42))),
                max_parallel_participants=mpp,
            )
            runner_fn.task_name = tname
            fns.append(runner_fn)
        elif tname == "factual_qa":
            c = task_cfg.get("factual_qa", {})
            if repeat is not None:
                n_participants_val = int(repeat)
            else:
                n_participants_val = int(c.get("n_participants", c.get("n_repeat", 1)))
            n_docs_val = c.get("n_docs")  # None = full pool; int = sample pool to that many docs
            runner_fn = lambda fn=fn, c=c, n_participants_val=n_participants_val, n_docs_val=n_docs_val, mpp=mpp: fn(
                llm, out_dir,
                model_cfg=model_cfg,
                data_json_path=c.get("data_json_path", "data/wikipedia_10docs_questions.json"),
                n_participants=n_participants_val,
                n_docs=int(n_docs_val) if n_docs_val is not None else None,
                stimuli_seed=int(c.get("stimuli_seed", run_cfg.get("seed", 42))),
                max_parallel_participants=mpp,
            )
            runner_fn.task_name = tname
            fns.append(runner_fn)
        elif tname == "narrative_qa":
            c = task_cfg.get("narrative_qa", {})
            if repeat is not None:
                n_participants_val = int(repeat)
            else:
                n_participants_val = int(c.get("n_participants", c.get("n_repeat", 1)))
            n_stories_val = c.get("n_stories")  # None = full pool
            runner_fn = lambda fn=fn, c=c, n_participants_val=n_participants_val, n_stories_val=n_stories_val, mpp=mpp: fn(
                llm, out_dir,
                model_cfg=model_cfg,
                data_json_path=c.get("data_json_path", "data/narrative_QA.json"),
                n_participants=n_participants_val,
                n_stories=int(n_stories_val) if n_stories_val is not None else None,
                stimuli_seed=int(c.get("stimuli_seed", run_cfg.get("seed", 42))),
                max_parallel_participants=mpp,
            )
            runner_fn.task_name = tname
            fns.append(runner_fn)
        elif tname == "map_task":
            c = task_cfg.get("map_task", {})
            if repeat is not None:
                n_participants_val = int(repeat)
            else:
                n_participants_val = int(c.get("n_participants", c.get("n_repeat", 1)))
            n_maps_val = c.get("n_maps")  # None = use all maps in file
            runner_fn = lambda fn=fn, c=c, n_participants_val=n_participants_val, n_maps_val=n_maps_val, mpp=mpp: fn(
                llm, out_dir,
                model_cfg=model_cfg,
                data_json_path=c.get("data_json_path", "data/maps.json"),
                n_participants=n_participants_val,
                n_maps=int(n_maps_val) if n_maps_val is not None else None,
                stimuli_seed=int(c.get("stimuli_seed", run_cfg.get("seed", 42))),
                max_parallel_participants=mpp,
            )
            runner_fn.task_name = tname
            fns.append(runner_fn)
        elif tname == "craft_task":
            c = task_cfg.get("craft_task", {})
            if repeat is not None:
                n_participants_val = int(repeat)
            else:
                n_participants_val = int(c.get("n_participants", c.get("n_repeat", 1)))
            n_tasks_val = c.get("n_tasks")  # None = all tasks in JSON (e.g. 15 questions total)
            runner_fn = lambda fn=fn, c=c, n_participants_val=n_participants_val, n_tasks_val=n_tasks_val, mpp=mpp: fn(
                llm, out_dir,
                model_cfg=model_cfg,
                data_json_path=c.get("data_json_path", "data/craft_task.json"),
                n_participants=n_participants_val,
                n_tasks=int(n_tasks_val) if n_tasks_val is not None else None,
                stimuli_seed=int(c.get("stimuli_seed", run_cfg.get("seed", 42))),
                max_parallel_participants=mpp,
            )
            runner_fn.task_name = tname
            fns.append(runner_fn)
        elif tname == "sum_semantic_story_recall":
            c = task_cfg.get(tname, task_cfg.get("wm_semantic_story_recall", {}))
            if repeat is not None:
                repeats_per_stimulus = int(repeat)
            else:
                repeats_per_stimulus = int(c.get("repeats_per_stimulus", 1))
            runner_fn = lambda fn=fn, c=c, _debug=debug, _story=story, mpp=mpp: fn(
                llm, out_dir,
                model_cfg=model_cfg,
                repeats_per_stimulus=repeats_per_stimulus,
                temperature=float(c.get("temperature", model_cfg.get("temperature", 0.0))),
                debug=_debug,
                story=_story,
                max_parallel_participants=mpp,
            )
            runner_fn.task_name = tname
            fns.append(runner_fn)
        elif tname == "sum_digit_span_forward":
            c = task_cfg.get(tname, task_cfg.get("wm_digit_span_forward", {}))
            if repeat is not None:
                sequences_per_span = 2 * int(repeat)
            else:
                sequences_per_span = int(c.get("sequences_per_span", 20))
            runner_fn = lambda fn=fn, c=c, _debug=debug, mpp=mpp: fn(
                llm, out_dir,
                model_cfg=model_cfg,
                stimuli_seed=int(c.get("stimuli_seed", run_cfg.get("seed", 42))),
                min_span=int(c.get("min_span", 2)),
                max_span=int(c.get("max_span", 20)),
                sequences_per_span=sequences_per_span,
                temperature=float(c.get("temperature", model_cfg.get("temperature", 0.0))),
                debug=_debug,
                max_parallel_participants=mpp,
            )
            runner_fn.task_name = tname
            fns.append(runner_fn)
        elif tname == "sum_digit_span_reverse":
            c = task_cfg.get(tname, task_cfg.get("wm_digit_span_reverse", {}))
            if repeat is not None:
                sequences_per_span = 2 * int(repeat)
            else:
                sequences_per_span = int(c.get("sequences_per_span", 20))
            runner_fn = lambda fn=fn, c=c, _debug=debug, mpp=mpp: fn(
                llm, out_dir,
                model_cfg=model_cfg,
                stimuli_seed=int(c.get("stimuli_seed", run_cfg.get("seed", 42))),
                min_span=int(c.get("min_span", 2)),
                max_span=int(c.get("max_span", 20)),
                sequences_per_span=sequences_per_span,
                temperature=float(c.get("temperature", model_cfg.get("temperature", 0.0))),
                debug=_debug,
                max_parallel_participants=mpp,
            )
            runner_fn.task_name = tname
            fns.append(runner_fn)
        elif tname == "sum_nback":
            c = task_cfg.get(tname, task_cfg.get("wm_nback", {}))
            if repeat is not None:
                n_repeat_val = int(repeat)
            else:
                n_repeat_val = int(c.get("n_repeat", 1))
            runner_fn = lambda fn=fn, c=c, _debug=debug, mpp=mpp: fn(
                llm, out_dir,
                model_cfg=model_cfg,
                n_repeat=n_repeat_val,
                n_repeats_per_participant=int(c.get("n_repeats_per_participant", 1)),
                stimuli_seed=int(c.get("stimuli_seed", 123)),
                temperature=float(c.get("temperature", model_cfg.get("temperature", 0.0))),
                debug=_debug,
                max_parallel_participants=mpp,
            )
            runner_fn.task_name = tname
            fns.append(runner_fn)
        elif tname == "sum_word_recognition":
            c = task_cfg.get(tname, task_cfg.get("wm_word_recognition", {}))
            if repeat is not None:
                n_repeat_val = int(repeat)
            else:
                n_repeat_val = int(c.get("n_repeat", 1))
            runner_fn = lambda fn=fn, c=c, _debug=debug, mpp=mpp: fn(
                llm, out_dir,
                model_cfg=model_cfg,
                words_json_path=c.get("words_json_path", "data/words.json"),
                n_repeat=n_repeat_val,
                max_trials_per_game=int(c.get("max_trials_per_game", 100)),
                stimuli_seed=int(c.get("stimuli_seed", run_cfg.get("seed", 42))),
                temperature=float(c.get("temperature", model_cfg.get("temperature", 0.0))),
                debug=_debug,
                max_parallel_participants=mpp,
            )
            runner_fn.task_name = tname
            fns.append(runner_fn)
        elif tname == "sum_variable_mapping":
            c = task_cfg.get(tname, task_cfg.get("wm_variable_mapping", {}))
            if repeat is not None:
                n_repeat_val = int(repeat)
            else:
                n_repeat_val = int(c.get("n_repeat", 1))
            runner_fn = lambda fn=fn, c=c, _debug=debug, mpp=mpp: fn(
                llm, out_dir,
                model_cfg=model_cfg,
                names_json_path=c.get("names_json_path", "data/names.json"),
                city_json_path=c.get("city_json_path", "data/city.json"),
                n_repeat=n_repeat_val,
                n_runs_per_participant=int(c.get("n_runs_per_participant", 3)),
                stimuli_seed=int(c.get("stimuli_seed", run_cfg.get("seed", 42))),
                temperature=float(c.get("temperature", model_cfg.get("temperature", 0.0))),
                debug=_debug,
                max_parallel_participants=mpp,
            )
            runner_fn.task_name = tname
            fns.append(runner_fn)
        elif tname == "sum_factual_qa":
            c = task_cfg.get(tname, task_cfg.get("wm_factual_qa", {}))
            if repeat is not None:
                n_participants_val = int(repeat)
            else:
                n_participants_val = int(c.get("n_participants", 1))
            n_docs_val = c.get("n_docs")
            runner_fn = lambda fn=fn, c=c, n_participants_val=n_participants_val, n_docs_val=n_docs_val, _debug=debug, mpp=mpp: fn(
                llm, out_dir,
                model_cfg=model_cfg,
                data_json_path=c.get("data_json_path", "data/wikipedia_10docs_questions.json"),
                n_participants=n_participants_val,
                n_docs=int(n_docs_val) if n_docs_val is not None else None,
                stimuli_seed=int(c.get("stimuli_seed", run_cfg.get("seed", 42))),
                temperature=float(c.get("temperature", model_cfg.get("temperature", 0.0))),
                debug=_debug,
                max_parallel_participants=mpp,
            )
            runner_fn.task_name = tname
            fns.append(runner_fn)
        elif tname == "sum_narrative_qa":
            c = task_cfg.get(tname, task_cfg.get("wm_narrative_qa", {}))
            if repeat is not None:
                n_participants_val = int(repeat)
            else:
                n_participants_val = int(c.get("n_participants", 1))
            n_stories_val = c.get("n_stories")
            runner_fn = lambda fn=fn, c=c, n_participants_val=n_participants_val, n_stories_val=n_stories_val, _debug=debug, mpp=mpp: fn(
                llm, out_dir,
                model_cfg=model_cfg,
                data_json_path=c.get("data_json_path", "data/narrative_QA.json"),
                n_participants=n_participants_val,
                n_stories=int(n_stories_val) if n_stories_val is not None else None,
                stimuli_seed=int(c.get("stimuli_seed", run_cfg.get("seed", 42))),
                temperature=float(c.get("temperature", model_cfg.get("temperature", 0.0))),
                debug=_debug,
                max_parallel_participants=mpp,
            )
            runner_fn.task_name = tname
            fns.append(runner_fn)
        elif tname == "sum_map_task":
            c = task_cfg.get(tname, task_cfg.get("wm_map_task", {}))
            if repeat is not None:
                n_participants_val = int(repeat)
            else:
                n_participants_val = int(c.get("n_participants", 1))
            n_maps_val = c.get("n_maps")
            runner_fn = lambda fn=fn, c=c, n_participants_val=n_participants_val, n_maps_val=n_maps_val, _debug=debug, mpp=mpp: fn(
                llm, out_dir,
                model_cfg=model_cfg,
                data_json_path=c.get("data_json_path", "data/maps.json"),
                n_participants=n_participants_val,
                n_maps=int(n_maps_val) if n_maps_val is not None else None,
                stimuli_seed=int(c.get("stimuli_seed", run_cfg.get("seed", 42))),
                temperature=float(c.get("temperature", model_cfg.get("temperature", 0.0))),
                debug=_debug,
                max_parallel_participants=mpp,
            )
            runner_fn.task_name = tname
            fns.append(runner_fn)
        elif tname == "sum_craft_task":
            c = task_cfg.get(tname, task_cfg.get("wm_craft_task", {}))
            if repeat is not None:
                n_participants_val = int(repeat)
            else:
                n_participants_val = int(c.get("n_participants", 1))
            n_tasks_val = c.get("n_tasks")
            runner_fn = lambda fn=fn, c=c, n_participants_val=n_participants_val, n_tasks_val=n_tasks_val, _debug=debug, mpp=mpp: fn(
                llm, out_dir,
                model_cfg=model_cfg,
                data_json_path=c.get("data_json_path", "data/craft_task.json"),
                n_participants=n_participants_val,
                n_tasks=int(n_tasks_val) if n_tasks_val is not None else None,
                stimuli_seed=int(c.get("stimuli_seed", run_cfg.get("seed", 42))),
                temperature=float(c.get("temperature", model_cfg.get("temperature", 0.0))),
                debug=_debug,
                max_parallel_participants=mpp,
            )
            runner_fn.task_name = tname
            fns.append(runner_fn)
        elif tname=="wm_semantic_story_recall":
            c = task_cfg.get("wm_semantic_story_recall", {})
            if repeat is not None:
                repeats_per_stimulus = int(repeat)
            else:
                repeats_per_stimulus = int(c.get("repeats_per_stimulus", 1))
            runner_fn = lambda fn=fn, c=c, _debug=debug, _story=story, mpp=mpp: fn(
                llm, out_dir,
                model_cfg=model_cfg,
                repeats_per_stimulus=repeats_per_stimulus,
                temperature=float(c.get("temperature", model_cfg.get("temperature", 0.0))),
                debug=_debug,
                story=_story,
                max_parallel_participants=mpp,
            )
            runner_fn.task_name = tname
            fns.append(runner_fn)
        elif tname == "wm_digit_span_forward":
            c = task_cfg.get("wm_digit_span_forward", {})
            if repeat is not None:
                sequences_per_span = 2 * int(repeat)
            else:
                sequences_per_span = int(c.get("sequences_per_span", 20))
            runner_fn = lambda fn=fn, c=c, _debug=debug: fn(
                llm, out_dir,
                model_cfg=model_cfg,
                stimuli_seed=int(c.get("stimuli_seed", run_cfg.get("seed", 42))),
                min_span=int(c.get("min_span", 2)),
                max_span=int(c.get("max_span", 20)),
                sequences_per_span=sequences_per_span,
                temperature=float(c.get("temperature", model_cfg.get("temperature", 0.0))),
                debug=_debug,
            )
            runner_fn.task_name = tname
            fns.append(runner_fn)
        elif tname == "wm_digit_span_reverse":
            c = task_cfg.get("wm_digit_span_reverse", {})
            if repeat is not None:
                sequences_per_span = 2 * int(repeat)
            else:
                sequences_per_span = int(c.get("sequences_per_span", 20))
            runner_fn = lambda fn=fn, c=c, _debug=debug: fn(
                llm, out_dir,
                model_cfg=model_cfg,
                stimuli_seed=int(c.get("stimuli_seed", run_cfg.get("seed", 42))),
                min_span=int(c.get("min_span", 2)),
                max_span=int(c.get("max_span", 20)),
                sequences_per_span=sequences_per_span,
                temperature=float(c.get("temperature", model_cfg.get("temperature", 0.0))),
                debug=_debug,
            )
            runner_fn.task_name = tname
            fns.append(runner_fn)
        elif tname == "wm_nback":
            c = task_cfg.get("wm_nback", {})
            if repeat is not None:
                n_repeat_val = int(repeat)
            else:
                n_repeat_val = int(c.get("n_repeat", 1))
            runner_fn = lambda fn=fn, c=c, _debug=debug, mpp=mpp: fn(
                llm, out_dir,
                model_cfg=model_cfg,
                n_repeat=n_repeat_val,
                n_repeats_per_participant=int(c.get("n_repeats_per_participant", 1)),
                stimuli_seed=int(c.get("stimuli_seed", 123)),
                temperature=float(c.get("temperature", model_cfg.get("temperature", 0.0))),
                debug=_debug,
                max_parallel_participants=mpp,
            )
            runner_fn.task_name = tname
            fns.append(runner_fn)
        elif tname == "wm_word_recognition":
            c = task_cfg.get("wm_word_recognition", {})
            if repeat is not None:
                n_repeat_val = int(repeat)
            else:
                n_repeat_val = int(c.get("n_repeat", 1))
            runner_fn = lambda fn=fn, c=c, _debug=debug, mpp=mpp: fn(
                llm, out_dir,
                model_cfg=model_cfg,
                words_json_path=c.get("words_json_path", "data/words.json"),
                n_repeat=n_repeat_val,
                max_trials_per_game=int(c.get("max_trials_per_game", 100)),
                stimuli_seed=int(c.get("stimuli_seed", run_cfg.get("seed", 42))),
                temperature=float(c.get("temperature", model_cfg.get("temperature", 0.0))),
                debug=_debug,
                max_parallel_participants=mpp,
            )
            runner_fn.task_name = tname
            fns.append(runner_fn)
        elif tname == "wm_variable_mapping":
            c = task_cfg.get("wm_variable_mapping", {})
            if repeat is not None:
                n_repeat_val = int(repeat)
            else:
                n_repeat_val = int(c.get("n_repeat", 1))
            runner_fn = lambda fn=fn, c=c, _debug=debug, mpp=mpp: fn(
                llm, out_dir,
                model_cfg=model_cfg,
                names_json_path=c.get("names_json_path", "data/names.json"),
                city_json_path=c.get("city_json_path", "data/city.json"),
                n_repeat=n_repeat_val,
                stimuli_seed=int(c.get("stimuli_seed", run_cfg.get("seed", 42))),
                temperature=float(c.get("temperature", model_cfg.get("temperature", 0.0))),
                debug=_debug,
                max_parallel_participants=mpp,
            )
            runner_fn.task_name = tname
            fns.append(runner_fn)
        elif tname == "wm_factual_qa":
            c = task_cfg.get("wm_factual_qa", {})
            if repeat is not None:
                n_participants_val = int(repeat)
            else:
                n_participants_val = int(c.get("n_participants", 1))
            n_docs_val = c.get("n_docs")
            runner_fn = lambda fn=fn, c=c, n_participants_val=n_participants_val, n_docs_val=n_docs_val, _debug=debug, mpp=mpp: fn(
                llm, out_dir,
                model_cfg=model_cfg,
                data_json_path=c.get("data_json_path", "data/wikipedia_10docs_questions.json"),
                n_participants=n_participants_val,
                n_docs=int(n_docs_val) if n_docs_val is not None else None,
                stimuli_seed=int(c.get("stimuli_seed", run_cfg.get("seed", 42))),
                temperature=float(c.get("temperature", model_cfg.get("temperature", 0.0))),
                debug=_debug,
                max_parallel_participants=mpp,
            )
            runner_fn.task_name = tname
            fns.append(runner_fn)
        elif tname == "wm_narrative_qa":
            c = task_cfg.get("wm_narrative_qa", {})
            if repeat is not None:
                n_participants_val = int(repeat)
            else:
                n_participants_val = int(c.get("n_participants", 1))
            n_stories_val = c.get("n_stories")
            runner_fn = lambda fn=fn, c=c, n_participants_val=n_participants_val, n_stories_val=n_stories_val, _debug=debug, mpp=mpp: fn(
                llm, out_dir,
                model_cfg=model_cfg,
                data_json_path=c.get("data_json_path", "data/narrative_QA.json"),
                n_participants=n_participants_val,
                n_stories=int(n_stories_val) if n_stories_val is not None else None,
                stimuli_seed=int(c.get("stimuli_seed", run_cfg.get("seed", 42))),
                temperature=float(c.get("temperature", model_cfg.get("temperature", 0.0))),
                debug=_debug,
                max_parallel_participants=mpp,
            )
            runner_fn.task_name = tname
            fns.append(runner_fn)
        elif tname == "wm_application_reading_qa":
            c = task_cfg.get("wm_application_reading_qa", {})
            if repeat is not None:
                n_participants_val = int(repeat)
            else:
                n_participants_val = int(c.get("n_participants", 1))
            n_docs_val = c.get("n_docs")
            runner_fn = lambda fn=fn, c=c, n_participants_val=n_participants_val, n_docs_val=n_docs_val, _debug=debug, mpp=mpp: fn(
                llm, out_dir,
                model_cfg=model_cfg,
                documents_dir=c.get("documents_dir", "application/documents"),
                n_participants=n_participants_val,
                n_docs=int(n_docs_val) if n_docs_val is not None else None,
                stimuli_seed=int(c.get("stimuli_seed", run_cfg.get("seed", 42))),
                temperature=float(c.get("temperature", model_cfg.get("temperature", 0.0))),
                debug=_debug,
                max_parallel_participants=mpp,
            )
            runner_fn.task_name = tname
            fns.append(runner_fn)
        elif tname == "wm_map_task":
            c = task_cfg.get("wm_map_task", {})
            if repeat is not None:
                n_participants_val = int(repeat)
            else:
                n_participants_val = int(c.get("n_participants", 1))
            n_maps_val = c.get("n_maps")
            runner_fn = lambda fn=fn, c=c, n_participants_val=n_participants_val, n_maps_val=n_maps_val, _debug=debug, mpp=mpp: fn(
                llm, out_dir,
                model_cfg=model_cfg,
                data_json_path=c.get("data_json_path", "data/maps.json"),
                n_participants=n_participants_val,
                n_maps=int(n_maps_val) if n_maps_val is not None else None,
                stimuli_seed=int(c.get("stimuli_seed", run_cfg.get("seed", 42))),
                temperature=float(c.get("temperature", model_cfg.get("temperature", 0.0))),
                debug=_debug,
                max_parallel_participants=mpp,
            )
            runner_fn.task_name = tname
            fns.append(runner_fn)
        elif tname == "wm_craft_task":
            c = task_cfg.get("wm_craft_task", {})
            if repeat is not None:
                n_participants_val = int(repeat)
            else:
                n_participants_val = int(c.get("n_participants", 1))
            n_tasks_val = c.get("n_tasks")
            runner_fn = lambda fn=fn, c=c, n_participants_val=n_participants_val, n_tasks_val=n_tasks_val, _debug=debug, mpp=mpp: fn(
                llm, out_dir,
                model_cfg=model_cfg,
                data_json_path=c.get("data_json_path", "data/craft_task.json"),
                n_participants=n_participants_val,
                n_tasks=int(n_tasks_val) if n_tasks_val is not None else None,
                stimuli_seed=int(c.get("stimuli_seed", run_cfg.get("seed", 42))),
                temperature=float(c.get("temperature", model_cfg.get("temperature", 0.0))),
                debug=_debug,
                max_parallel_participants=mpp,
            )
            runner_fn.task_name = tname
            fns.append(runner_fn)
        else:
            runner_fn = lambda fn=fn: fn(llm, out_dir, model_cfg=model_cfg)
            runner_fn.task_name = tname
            fns.append(runner_fn)

    mpt = run_cfg.get("max_parallel_tasks")

    t0 = time.perf_counter()
    overall = run_tasks(fns, out_dir, max_parallel_tasks=int(mpt) if mpt is not None else None)
    elapsed_s = time.perf_counter() - t0
    sys.stderr.write("\n")
    sys.stderr.flush()
    typer.echo(
        f"Done. Wall time: {elapsed_s:.1f}s | LLM API requests: {getattr(llm, 'request_count', 0)}"
    )

    # Print + persist metrics tables.
    # One experiment/task -> one table (rows are conditions C1..C4).
    from .core.tables import render_text_table, write_metrics_markdown

    saved_paths = []
    for task_summary in overall.get("tasks", []):
        task_name = task_summary.get("task", "unknown_task")
        conds = task_summary.get("conditions", [])

        rows = []
        for c in conds:
            row = {"condition": c.get("condition", "")}
            if "n" in c:
                row["n"] = c["n"]
            row.update(c.get("metrics", {}))
            rows.append(row)

        table_str = render_text_table(rows)
        txt_path = out_dir / f"metrics_{task_name}.txt"
        txt_path.write_text(table_str + "\n", encoding="utf-8")
        saved_paths.append(str(txt_path))

        typer.echo("\n" + "=" * 60)
        typer.echo(task_name)
        typer.echo(table_str)
        typer.echo(f"Saved: {txt_path}")

    # Also write an index file listing all per-task metric tables.
    (out_dir / "metrics_index.txt").write_text("\n".join(saved_paths) + "\n", encoding="utf-8")
    typer.echo(f"\nSaved: {out_dir / 'metrics_index.txt'}")
    md_path = out_dir / "metrics_all_tasks.md"
    write_metrics_markdown(overall.get("tasks", []), md_path)
    typer.echo(f"Saved: {md_path}")
    typer.echo(yaml.safe_dump(overall, sort_keys=False))


@app.command()
def tables(
    run_dir: str = typer.Argument(..., help="Run directory (e.g., runs/exp1)"),
):
    """Generate CSV + HTML tables from an existing run's summary.json."""
    from .core.io import read_json
    from .core.tables import flatten_metrics, write_metrics_csv, write_metrics_html, write_metrics_markdown

    rd = Path(run_dir)
    summary = read_json(rd/"summary.json")
    rows = flatten_metrics(summary.get("tasks", []))
    write_metrics_csv(rows, rd/"tables"/"metrics.csv")
    write_metrics_html(rows, rd/"tables"/"metrics.html")
    write_metrics_markdown(summary.get("tasks", []), rd/"tables"/"metrics_all_tasks.md")
    typer.echo(str(rd/"tables"/"metrics.csv"))
    typer.echo(str(rd/"tables"/"metrics.html"))
    typer.echo(str(rd/"tables"/"metrics_all_tasks.md"))

if __name__ == "__main__":
    app()
