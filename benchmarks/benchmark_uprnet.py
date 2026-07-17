from __future__ import annotations

from benchmarks.common import run_benchmark


def main(argv: list[str] | None = None) -> None:
    run_benchmark(
        default_config="configs/run/inference_uprnet_official.yaml",
        model_label="UPRNet",
        argv=argv,
    )


if __name__ == "__main__":
    main()
