from __future__ import annotations

from benchmarks.common import run_benchmark


def main(argv: list[str] | None = None) -> None:
    run_benchmark(
        default_config="configs/run/inference_emavfi_official.yaml",
        model_label="EMAVFI",
        argv=argv,
    )


if __name__ == "__main__":
    main()
