from pathlib import Path
import argparse
import time
from typing import Tuple

import os
os.environ.setdefault("PYTORCH_SILENCE_ALLOW_TF32_DEPRECATION", "1")

import torch
from torch.utils.data import DataLoader

from datasets.van_der_pol import VanDerPolDataset
from function_encoder.model.mlp import MLP
from function_encoder.model.neural_ode import NeuralODE, ODEFunc, rk4_step
from function_encoder.function_encoder import BasisFunctions, FunctionEncoder


def configure_tf32(device: str, matmul_precision: str = "high", conv_precision: str = "tf32") -> None:
    """Set TF32 preferences using the newest available PyTorch API."""
    if device != "cuda":
        return

    precision_aliases = {
        "high": "ieee",
        "medium": "tf32",
        "low": "tf32",
    }
    normalized_matmul = precision_aliases.get(matmul_precision, matmul_precision)
    allowed_precisions = {"ieee", "tf32", "none"}
    if normalized_matmul not in allowed_precisions:
        normalized_matmul = "tf32"

    matmul_backend = getattr(torch.backends.cuda, "matmul", None)
    if matmul_backend is not None and hasattr(matmul_backend, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = normalized_matmul

    cudnn_backend = getattr(torch.backends, "cudnn", None)
    conv_backend = getattr(cudnn_backend, "conv", None) if cudnn_backend else None
    if conv_backend is not None and hasattr(conv_backend, "fp32_precision"):
        torch.backends.cudnn.conv.fp32_precision = conv_precision


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def build_model(n_basis: int, device: str) -> FunctionEncoder:
    basis_functions = BasisFunctions(
        *[
            NeuralODE(
                ode_func=ODEFunc(model=MLP(layer_sizes=[3, 64, 64, 2])),
                integrator=rk4_step,
            )
            for _ in range(n_basis)
        ]
    )
    model = FunctionEncoder(basis_functions).to(device)
    return model


@torch.no_grad()
def compute_coefficients_for_batch(
    model: FunctionEncoder,
    batch: Tuple[torch.Tensor, ...],
    device: str,
) -> torch.Tensor:
    _, _, _, _, y0_example, dt_example, y1_example = batch
    y0_example = y0_example.to(device)
    dt_example = dt_example.to(device)
    y1_example = y1_example.to(device)
    coefficients, _ = model.compute_coefficients((y0_example, dt_example), y1_example)
    return coefficients


def rollout_baseline_per_trajectory(
    model: FunctionEncoder,
    x0: torch.Tensor,  # [B, 2]
    dt: torch.Tensor,  # [B]
    coefficients: torch.Tensor,  # [B, K]
    n_steps: int,
) -> torch.Tensor:
    model.eval()
    B, D = x0.shape
    traj = torch.empty(B, n_steps + 1, D, device=x0.device, dtype=x0.dtype)
    traj[:, 0] = x0
    for i in range(B):
        x = x0[i].unsqueeze(0).unsqueeze(1)  # [1,1,D]
        dt_i = dt[i].unsqueeze(0).unsqueeze(0)  # [1,1]
        c_i = coefficients[i].unsqueeze(0)  # [1,K]
        for k in range(n_steps):
            dx = model((x, dt_i), coefficients=c_i)
            x = x + dx
            traj[i, k + 1] = x.squeeze(1)
    return traj


def rollout_batched(
    model: FunctionEncoder,
    x0: torch.Tensor,
    dt: torch.Tensor,
    coefficients: torch.Tensor,
    n_steps: int,
) -> torch.Tensor:
    model.eval()
    B, D = x0.shape
    x = x0.unsqueeze(1)  # [B,1,D]
    dt_b = dt.unsqueeze(1)  # [B,1]
    traj = torch.empty(B, n_steps + 1, D, device=x0.device, dtype=x0.dtype)
    traj[:, 0] = x0
    for k in range(n_steps):
        dx = model((x, dt_b), coefficients=coefficients)
        x = x + dx
        traj[:, k + 1] = x.squeeze(1)
    return traj


def make_rollout_fn(vectorized: bool, compiled: bool, n_steps: int):
    if vectorized:
        def rollout(model, x0, dt, coefficients):
            return rollout_batched(model, x0, dt, coefficients, n_steps)
    else:
        def rollout(model, x0, dt, coefficients):
            return rollout_baseline_per_trajectory(model, x0, dt, coefficients, n_steps)

    if compiled:
        rollout = torch.compile(rollout, mode="reduce-overhead")
    return rollout


def time_fn(fn, warmup: int, runs: int) -> float:
    # Warmup
    for _ in range(warmup):
        _ = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # Timed runs
    start = time.perf_counter()
    for _ in range(runs):
        _ = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) / runs


def main():
    parser = argparse.ArgumentParser(description="Compare inference speed: baseline vs vectorized+compiled")
    parser.add_argument("--batch_size", type=int, default=9, help="Number of trajectories to roll out")
    parser.add_argument("--step_size", type=float, default=0.1, help="Fixed dt for rollout")
    parser.add_argument("--horizon", type=float, default=10.0, help="Rollout horizon (seconds)")
    parser.add_argument("--runs", type=int, default=100, help="Number of timed runs to average")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup runs")
    parser.add_argument("--n_basis", type=int, default=10, help="Number of basis functions")
    parser.add_argument("--weights", type=str, default="results/van_der_pol/van_der_pol_model.pth", help="Path to model weights")
    all_modes = ["scalar_eager", "scalar_compiled", "vectorized_eager", "vectorized_compiled"]
    parser.add_argument(
        "--modes",
        nargs="+",
        default=all_modes,
        choices=all_modes,
        help="Which rollout configurations to benchmark",
    )
    args = parser.parse_args()

    torch.manual_seed(42)

    device = get_device()
    configure_tf32(device)
    model = build_model(n_basis=args.n_basis, device=device)

    weights_path = Path(args.weights)
    if weights_path.exists():
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
    else:
        print(f"Warning: weights not found at {weights_path}, running with random initialization.")

    # Prepare a single batch for coefficient computation
    dataset = VanDerPolDataset(n_points=1000, n_example_points=100, dt_range=(0.1, 0.1))
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    batch = next(iter(dataloader))

    model.eval()
    with torch.no_grad():
        coefficients = compute_coefficients_for_batch(model, batch, device)  # [B, K]

    # Prepare rollout initial conditions (same distribution as dataset)
    y0_range = dataset.y0_range
    B = args.batch_size
    D = 2
    s = args.step_size
    n_steps = int(args.horizon / s)

    x0 = torch.empty(B, D, device=device).uniform_(*y0_range)
    dt = torch.full((B,), s, device=device)

    mode_specs = {
        "scalar_eager": {"vectorized": False, "compiled": False, "label": "Scalar + eager"},
        "scalar_compiled": {"vectorized": False, "compiled": True, "label": "Scalar + torch.compile"},
        "vectorized_eager": {"vectorized": True, "compiled": False, "label": "Vectorized + eager"},
        "vectorized_compiled": {"vectorized": True, "compiled": True, "label": "Vectorized + torch.compile"},
    }

    timings = []
    print("Inference timing (averaged):")
    for mode_name in args.modes:
        spec = mode_specs[mode_name]
        rollout = make_rollout_fn(
            vectorized=spec["vectorized"], compiled=spec["compiled"], n_steps=n_steps
        )

        def runner():
            with torch.no_grad():
                return rollout(model, x0, dt, coefficients)

        timing = time_fn(runner, warmup=args.warmup, runs=args.runs)
        timings.append((mode_name, spec["label"], timing))
        print(f"- {spec['label']:<30}: {timing * 1e3:.2f} ms/run")

    if {"scalar_eager", "vectorized_compiled"}.issubset(set(args.modes)):
        scalar = next(t for t in timings if t[0] == "scalar_eager")[2]
        vectorized_compiled = next(t for t in timings if t[0] == "vectorized_compiled")[2]
        if vectorized_compiled > 0:
            print(f"- Speedup (scalar eager -> vectorized compiled): {scalar / vectorized_compiled:.2f}x")


if __name__ == "__main__":
    main()
