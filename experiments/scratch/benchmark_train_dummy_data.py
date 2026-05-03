from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import torch
from torch.optim import AdamW

PROJECT_ROOT: Path = Path(__file__).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import forward_model
from src.models.registry import get_model_class


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark one training step loop with dummy data tensors already on device.")
    parser.add_argument("--model-name", choices=["IFRNet", "IFRNet_Residual"], default="IFRNet")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    return parser


def build_dummy_batch(
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    img0 = torch.rand((batch_size, 3, height, width), device=device, dtype=torch.float32)
    imgt = torch.rand((batch_size, 3, height, width), device=device, dtype=torch.float32)
    img1 = torch.rand((batch_size, 3, height, width), device=device, dtype=torch.float32)
    bmv = torch.rand((batch_size, 2, height, width), device=device, dtype=torch.float32)
    fmv = torch.rand((batch_size, 2, height, width), device=device, dtype=torch.float32)
    embt = torch.full((batch_size, 1, 1, 1), 0.5, device=device, dtype=torch.float32)
    return img0, imgt, img1, bmv, fmv, embt


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_class = get_model_class(args.model_name)
    model = model_class().to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    batch = build_dummy_batch(args.batch_size, args.height, args.width, device)

    step_durations: list[float] = []
    batch_size = args.batch_size

    for step_index in range(1, args.steps + 1):
        img0, imgt, img1, bmv, fmv, embt = batch

        synchronize_device(device)
        step_start = time.perf_counter()
        optimizer.zero_grad()
        imgt_pred, loss_rec, loss_geo, loss_dis, _flow0, _flow1, _mask = forward_model(
            args.model_name,
            model,
            img0,
            img1,
            embt,
            imgt,
            bmv,
            fmv,
        )
        total_loss = loss_rec + loss_geo + loss_dis
        total_loss.backward()
        optimizer.step()
        synchronize_device(device)
        step_end = time.perf_counter()

        step_duration = step_end - step_start
        if step_index > args.warmup_steps:
            step_durations.append(step_duration)

        if step_index == 1 or step_index == args.warmup_steps or step_index == args.steps:
            print(
                f"step={step_index} "
                f"step_seconds={step_duration:.6f} "
                f"imgt_pred_shape={tuple(imgt_pred.shape)} "
                f"loss_total={float(total_loss.detach().cpu()):.6f}"
            )

    if len(step_durations) == 0:
        raise RuntimeError("No benchmark steps were recorded. Increase --steps or decrease --warmup-steps.")

    average_step_seconds = sum(step_durations) / len(step_durations)
    samples_per_second = batch_size / average_step_seconds

    print(f"device={device}")
    print(f"model_name={args.model_name}")
    print(f"batch_size={args.batch_size}")
    print(f"resolution={args.height}x{args.width}")
    print(f"measured_steps={len(step_durations)}")
    print(f"average_step_seconds={average_step_seconds:.6f}")
    print(f"samples_per_second={samples_per_second:.2f}")


if __name__ == "__main__":
    main()
