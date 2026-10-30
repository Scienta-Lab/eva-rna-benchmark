"""Minimal local Metaflow entry point."""

import os
import sys
from pathlib import Path

os.environ.setdefault("METAFLOW_DEFAULT_DATASTORE", "local")
os.environ.setdefault("METAFLOW_DEFAULT_METADATA", "local")

repository_root = Path(__file__).resolve().parent.parent
if str(repository_root) not in sys.path:
    sys.path.insert(0, str(repository_root))

from metaflow import FlowSpec, Parameter, step  # noqa: E402

from benchmark.runner import run_task  # noqa: E402


class EVABenchmarkFlow(FlowSpec):
    """Run one task, model, and seed as a Metaflow execution."""

    benchmark_task = Parameter("benchmark-task", required=True)
    model = Parameter("model", required=True)
    checkpoint = Parameter("checkpoint", default="")
    data_dir = Parameter("data-dir", default="datasets")
    results_dir = Parameter("results-dir", default="runs")
    seed = Parameter("seed", default=10, type=int)
    device = Parameter("device", default="cpu")
    vocab_map = Parameter("vocab-map", default="")
    batch_size = Parameter("batch-size", type=int, default=0)
    result_model_name = Parameter("result-model-name", default="")
    tensorboard = Parameter("tensorboard", type=bool, default=False)

    @step
    def start(self) -> None:
        """Execute the selected benchmark task."""
        self.result_path = str(
            run_task(
                task_id=self.benchmark_task,
                model_name=self.model,
                checkpoint=self.checkpoint,
                data_dir=self.data_dir,
                results_dir=self.results_dir,
                seed=self.seed,
                device=self.device,
                vocab_map=self.vocab_map or None,
                batch_size=self.batch_size or None,
                result_model_name=self.result_model_name or None,
                tensorboard=self.tensorboard,
            )
        )
        self.next(self.end)

    @step
    def end(self) -> None:
        """Expose the generated result path as a Metaflow artifact."""
        print(self.result_path)


if __name__ == "__main__":
    EVABenchmarkFlow()
