from __future__ import annotations
from pathlib import Path
from typing import Optional

import typer
import yaml

from bench.core.io import ensure_dir, write_json
from bench.core.llm_openai import OpenAIChatLLM

from .evaluate import evaluate
from .plotting import plot_from_summary


app = typer.Typer(add_completion=False)


@app.callback()
def main() -> None:
    """Reading QA runner."""


@app.command()
def run(
    model: str = typer.Option("openai/gpt-4.1-mini", "--model", help="Model id."),
    backend: str = typer.Option("openai", "--backend", help="openai or anthropic"),
    n_repeat: int = typer.Option(1, "--n-repeat", min=1, help="Number of repeats."),
    seed: int = typer.Option(42, "--seed", help="Sampling seed."),
    documents_dir: str = typer.Option(
        "application/documents",
        "--documents-dir",
        help="Directory with document json files.",
    ),
    out_dir: Optional[str] = typer.Option(
        None,
        "--out-dir",
        help="Output directory. Default: application/out/<model_slug>",
    ),
    temperature: float = typer.Option(0.0, "--temperature"),
    max_tokens: int = typer.Option(2048, "--max-tokens"),
    top_p: float = typer.Option(1.0, "--top-p"),
    base_url: Optional[str] = typer.Option(None, "--base-url"),
    qwen_thinking: Optional[str] = typer.Option(
        None,
        "--qwen-thinking",
        help='For qwen/qwen3-8b only: "true" or "false" to set enable_thinking.',
    ),
    max_parallel_repeats: Optional[int] = typer.Option(
        None,
        "--max-parallel-repeats",
        min=1,
        help="Max number of repeats to run concurrently. Default uses n_repeat.",
    ),
):
    model_cfg = {
        "name": model,
        "backend": backend,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "top_p": float(top_p),
    }

    extra_body = None
    model_name_l = str(model).lower()
    if qwen_thinking is not None:
        if "qwen/qwen3-8b" in model_name_l:
            v = qwen_thinking.strip().lower()
            if v in ("true", "1", "yes"):
                extra_body = {"enable_thinking": True}
            elif v in ("false", "0", "no"):
                extra_body = {"enable_thinking": False}
            else:
                raise typer.BadParameter('--qwen-thinking must be "true" or "false".')
        else:
            typer.secho(
                f'Warning: --qwen-thinking ignored for model "{model}".',
                err=True,
            )

    model_slug = model.replace("/", "_").replace("\\", "_")
    if isinstance(extra_body, dict) and isinstance(extra_body.get("enable_thinking"), bool):
        model_slug = f"{model_slug}_thinking_{'true' if extra_body['enable_thinking'] else 'false'}"
    run_out = Path(out_dir) if out_dir else Path("application/out") / model_slug
    ensure_dir(run_out)

    backend_l = backend.lower()
    if backend_l == "anthropic":
        from bench.core.llm_anthropic import AnthropicChatLLM

        llm = AnthropicChatLLM(
            model=model_cfg["name"],
            base_url=base_url,
        )
    else:
        llm = OpenAIChatLLM(
            model=model_cfg["name"],
            base_url=base_url,
            extra_body=extra_body,
        )

    summary = evaluate(
        llm,
        run_out,
        model_cfg=model_cfg,
        documents_dir=Path(documents_dir),
        n_repeat=n_repeat,
        seed=seed,
        max_tokens=max_tokens,
        max_parallel_repeats=max_parallel_repeats,
    )

    write_json(
        run_out / "run_config.json",
        {
            "model": model_cfg,
            "n_repeat": n_repeat,
            "seed": seed,
            "documents_dir": documents_dir,
            "out_dir": str(run_out),
            "extra_body": extra_body or {},
        },
    )
    typer.echo(yaml.safe_dump(summary, sort_keys=False))
    typer.echo(f"Saved: {run_out / 'tasks' / 'application_reading_qa.jsonl'}")
    typer.echo(f"Saved: {run_out / 'tasks' / 'application_reading_qa_summary.json'}")


@app.command()
def plot(
    summary_json: str = typer.Option(
        ...,
        "--summary-json",
        help="Path to application_reading_qa_summary.json",
    ),
    out_dir: Optional[str] = typer.Option(
        None,
        "--out-dir",
        help="Output directory for figures. Default: sibling figures/ folder",
    ),
):
    summary_path = Path(summary_json)
    figures_dir = Path(out_dir) if out_dir else summary_path.parent / "figures"
    paths = plot_from_summary(summary_path, figures_dir)
    for p in paths:
        typer.echo(f"Saved: {p}")


if __name__ == "__main__":
    app()
