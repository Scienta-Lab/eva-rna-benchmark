"""Run many benchmark tasks for one model.

A full reproduction is one execution per task, model and seed -- 130 per model
across the four manifest families, with ZSP run separately. This drives that
sweep from the task manifest so it does not have to be scripted by hand, and
skips tasks a model cannot run rather than failing the sweep partway through.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from benchmark.runner import result_file_name, run_task


def planned_runs(
    data_dir: Path,
    model_name: str,
    families: set[str] | None,
    task_ids: set[str] | None,
) -> list[tuple[str, int]]:
    """List the (task, seed) pairs this model is declared to support."""
    tasks = pd.read_csv(data_dir / "tasks.csv", dtype=str).fillna("")
    supported = tasks[
        tasks["models"].str.split("|").apply(lambda names: model_name in names)
    ]
    if families:
        supported = supported[supported["family"].isin(families)]
    if task_ids:
        supported = supported[supported["task_id"].isin(task_ids)]
    return [
        (task.task_id, int(seed))
        for task in supported.itertuples(index=False)
        for seed in task.seeds.split("|")
    ]


def run_sweep(
    *,
    model_name: str,
    checkpoint: str,
    data_dir: Path,
    results_dir: Path,
    device: str = "cpu",
    vocab_map: str | None = None,
    batch_size: int | None = None,
    families: set[str] | None = None,
    task_ids: set[str] | None = None,
) -> list[Path]:
    """Run every planned task and seed in order, returning the files written.

    Runs are sequential because each one merges its rows into a shared result
    file; concurrent writers would interleave. Shard with `--task` across
    processes when that matters, giving each shard its own `--results-dir`.
    """
    runs = planned_runs(data_dir, model_name, families, task_ids)
    if not runs:
        raise ValueError(f"No released task lists {model_name!r} as a supported model")
    name = result_file_name(model_name, checkpoint, None)
    print(f"{len(runs)} runs for {name}", flush=True)
    written: list[Path] = []
    for index, (task_id, seed) in enumerate(runs, start=1):
        print(f"  [{index}/{len(runs)}] {task_id} seed {seed}", flush=True)
        written.append(
            run_task(
                task_id=task_id,
                model_name=model_name,
                checkpoint=checkpoint,
                data_dir=data_dir,
                results_dir=results_dir,
                seed=seed,
                device=device,
                vocab_map=vocab_map,
                batch_size=batch_size,
            )
        )
    return sorted(set(written))


def main() -> None:
    """Run a sweep from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets"))
    parser.add_argument("--results-dir", type=Path, default=Path("runs"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--vocab-map")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--family",
        action="append",
        help="Restrict to a task family; repeat to select several",
    )
    parser.add_argument(
        "--task",
        action="append",
        help="Restrict to a task id; repeat to select several",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the runs that would happen and exit",
    )
    arguments = parser.parse_args()
    families = set(arguments.family) if arguments.family else None
    task_ids = set(arguments.task) if arguments.task else None
    if arguments.dry_run:
        runs = planned_runs(arguments.data_dir, arguments.model, families, task_ids)
        for task_id, seed in runs:
            print(f"{task_id} seed {seed}")
        print(f"{len(runs)} runs")
        return
    for path in run_sweep(
        model_name=arguments.model,
        checkpoint=arguments.checkpoint,
        data_dir=arguments.data_dir,
        results_dir=arguments.results_dir,
        device=arguments.device,
        vocab_map=arguments.vocab_map,
        batch_size=arguments.batch_size,
        families=families,
        task_ids=task_ids,
    ):
        print(path)


if __name__ == "__main__":
    main()
