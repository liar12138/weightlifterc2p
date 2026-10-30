"""Grid-search depth noise/blur parameters against a real depth recording."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.v2 as v2
import tyro

#nohup python -u scripts/grid_search_depth_noise.py > logs/depth/grid/grid_search.log 2>&1 &
@dataclass(frozen=True)
class Args:
  sim_path: str = "logs/depth/sim/sim_depth_h2_2_2_30_60.npy"
  real_path: str = "logs/depth/real/filtered_depth_30_60.npy"
  output_csv: str = "logs/depth/depth_noise_grid_search_30_60.csv"
  output_best_npy: str | None = "logs/depth/grid/sim_depth_h2_grid_best_30_60.npy"
  height: int = 27
  width: int = 48
  max_depth: float = 6.0
  invalid_threshold: float = 1.0e-4
  frame_stride: int = 4
  seeds: tuple[int, ...] = (0, 1, 2)
  noise_means: tuple[float, ...] = (-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10)
  noise_sigmas: tuple[float, ...] = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05)
  blur_kernels: tuple[int, ...] = (1, 3, 5)
  blur_sigmas: tuple[float, ...] = (0.3, 0.5, 0.8, 1.0, 1.5)
  device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
  top_k: int = 20


def _load_depth(path: str, height: int, width: int) -> np.ndarray:
  data = np.load(path).astype(np.float32, copy=False)
  if data.ndim == 2 and data.shape[1] == height * width:
    data = data.reshape(-1, height, width)
  if data.ndim != 3 or data.shape[1:] != (height, width):
    raise ValueError(
      f"Expected {path} to have shape [T,{height},{width}] or "
      f"[T,{height * width}], got {data.shape}"
    )
  return data


def _replace_invalid(
    depth: torch.Tensor,
    threshold: float,
    max_depth: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    invalid = (
        ~torch.isfinite(depth)
        | (depth <= threshold)
        | (depth >= max_depth)
    )

    output = torch.where(
        invalid,
        torch.full_like(depth, max_depth),
        depth,
    )

    return output, invalid


def _apply_noise(
    depth: torch.Tensor,
    noise_mean: float,
    noise_sigma: float,
    blur_kernel: int,
    blur_sigma: float,
    max_depth: float,
    invalid_mask: torch.Tensor,
) -> torch.Tensor:
    output = v2.GaussianNoise(
        mean=noise_mean,
        sigma=noise_sigma,
        clip=False,
    )(depth)

    if blur_kernel > 1:
        output = v2.GaussianBlur(
            kernel_size=blur_kernel,
            sigma=blur_sigma,
        )(output)

    output = output.clamp(
        min=0.001,
        max=max_depth,
    )

    # 加噪和模糊后，无效位置重新强制设为 6 m
    output = torch.where(
        invalid_mask,
        torch.full_like(output, max_depth),
        output,
    )

    return output


def _valid_blur_pairs(args: Args):
  yield 1, 0.0
  for kernel in args.blur_kernels:
    if kernel <= 1:
      continue
    for sigma in args.blur_sigmas:
      yield kernel, sigma


def main(args: Args) -> None:
  sim_np = _load_depth(args.sim_path, args.height, args.width)
  real_np = _load_depth(args.real_path, args.height, args.width)
  num_frames = min(len(sim_np), len(real_np))
  if len(sim_np) != len(real_np):
    print(f"[WARN] Frame counts differ: sim={len(sim_np)}, real={len(real_np)}")
  sim_np = sim_np[:num_frames:args.frame_stride]
  real_np = real_np[:num_frames:args.frame_stride]

  sim = torch.from_numpy(sim_np).unsqueeze(1).to(args.device)
  real = torch.from_numpy(real_np).unsqueeze(1).to(args.device)
  sim ,sim_invalid= _replace_invalid(sim, args.invalid_threshold, args.max_depth)

  real_mean = real.mean()
  real_var = real.var(unbiased=False)
  real_std = real_var.sqrt()
  print(
    f"Search frames={len(sim_np)}, real mean={real_mean.item():.6f}, "
    f"real std={real_std.item():.6f}, real var={real_var.item():.6f}"
  )

  results: list[dict[str, float | int]] = []
  with torch.no_grad():
    for noise_mean in args.noise_means:
      for noise_sigma in args.noise_sigmas:
        for blur_kernel, blur_sigma in _valid_blur_pairs(args):
          seed_metrics = []
          for seed in args.seeds:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
              torch.cuda.manual_seed_all(seed)
            candidate = _apply_noise(
              sim,
              noise_mean,
              noise_sigma,
              blur_kernel,
              blur_sigma,
              args.max_depth,
              sim_invalid,
            )
            mean = candidate.mean()
            var = candidate.var(unbiased=False)
            mean_error = (mean - real_mean).abs()
            var_error = (var - real_var).abs()
            # Dimensionless score so meter and meter^2 errors are comparable.
            score = (mean_error / real_std).square() + (
              var_error / real_var.clamp_min(1e-12)
            ).square()
            difference = candidate - real
            seed_metrics.append(
              (
                score.item(),
                mean.item(),
                var.item(),
                mean_error.item(),
                var_error.item(),
                difference.abs().mean().item(),
                difference.square().mean().sqrt().item(),
              )
            )
          averaged = np.asarray(seed_metrics).mean(axis=0)
          results.append(
            {
              "score": averaged[0],
              "noise_mean": noise_mean,
              "noise_sigma": noise_sigma,
              "blur_kernel": blur_kernel,
              "blur_sigma": blur_sigma,
              "sim_mean": averaged[1],
              "sim_var": averaged[2],
              "mean_abs_error": averaged[3],
              "variance_abs_error": averaged[4],
              "pixel_mae": averaged[5],
              "pixel_rmse": averaged[6],
            }
          )

  results.sort(key=lambda item: item["score"])
  output_csv = Path(args.output_csv)
  output_csv.parent.mkdir(parents=True, exist_ok=True)
  with output_csv.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

  print(f"\nTop {min(args.top_k, len(results))} parameter sets:")
  for rank, result in enumerate(results[: args.top_k], start=1):
    print(
      f"{rank:2d}. score={result['score']:.8f}, "
      f"noise=({result['noise_mean']:.3f}, {result['noise_sigma']:.3f}), "
      f"blur=({result['blur_kernel']}, {result['blur_sigma']:.2f}), "
      f"mean={result['sim_mean']:.6f}, var={result['sim_var']:.6f}, "
      f"pixel_mae={result['pixel_mae']:.6f}"
    )
  print(f"\nFull results saved to {output_csv}")

  if args.output_best_npy is not None:
    best = results[0]
    full_sim = torch.from_numpy(
      _load_depth(args.sim_path, args.height, args.width)
    ).unsqueeze(1).to(args.device)
    full_sim ,full_sim_invalid= _replace_invalid(full_sim, args.invalid_threshold, args.max_depth)
    torch.manual_seed(args.seeds[0])
    best_depth = _apply_noise(
      full_sim,
      float(best["noise_mean"]),
      float(best["noise_sigma"]),
      int(best["blur_kernel"]),
      float(best["blur_sigma"]),
      args.max_depth,
      full_sim_invalid,
    )
    output_best = Path(args.output_best_npy)
    output_best.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_best, best_depth.squeeze(1).cpu().numpy())
    print(f"Best transformed depth saved to {output_best}")


if __name__ == "__main__":
  main(tyro.cli(Args))
