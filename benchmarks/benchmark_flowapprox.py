from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.common import run_benchmark


def main(argv: list[str] | None = None) -> None:
    run_benchmark(
        default_config="configs/run/inference_flowaprox_layer_1_0618.yaml",
        model_label="IFRNet_Residual_FlowApprox",
        argv=argv,
    )


if __name__ == "__main__":
    main()
