# -*- coding: utf-8 -*-
"""Standalone ONNX exporter for Qmini rsl_rl checkpoints.

Reconstructs the actor MLP directly from the checkpoint's ``actor_state_dict``
and exports it to ONNX. It does NOT boot Isaac Sim and does not depend on the
rsl_rl version, so it works with the newer ``mlp.*`` weight layout that the
built-in Isaac Lab exporter (in scripts/rsl_rl/play.py) cannot trace.

The Qmini PPO config uses no observation normalization, so the exported graph
maps raw observations directly to action means (the policy's deterministic
output used at deployment time).

Usage:
    # Pass a date folder; the latest model_*.pt under logs/rsl_rl/qmini/<date>/ is exported.
    python scripts/export_onnx.py 2026-05-24

The ONNX is always written to <project_root>/policy.onnx.
"""

import argparse
import glob
import os
import re
from collections import OrderedDict

import torch
import torch.nn as nn

# Training runs live at <project_root>/logs/rsl_rl/qmini/<date>/model_*.pt
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LOG_ROOT = os.path.join(PROJECT_ROOT, "logs", "rsl_rl", "qmini")

ACTIVATIONS = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
}


def build_mlp_from_state_dict(state_dict: "OrderedDict[str, torch.Tensor]", activation: str) -> nn.Sequential:
    """Rebuild the actor MLP from ``mlp.<idx>.{weight,bias}`` entries.

    The layer indices encode the original Sequential layout (Linear at even
    indices, activation at odd indices), so we replay that structure exactly.
    """
    act_cls = ACTIVATIONS[activation]
    # Collect Linear layers in order: mlp.0, mlp.2, mlp.4, ...
    linear_indices = sorted(
        {int(k.split(".")[1]) for k in state_dict if k.startswith("mlp.") and k.endswith(".weight")}
    )
    if not linear_indices:
        raise ValueError("No 'mlp.*.weight' entries found in actor_state_dict.")

    layers: list[nn.Module] = []
    for i, idx in enumerate(linear_indices):
        w = state_dict[f"mlp.{idx}.weight"]
        out_features, in_features = w.shape
        layers.append(nn.Linear(in_features, out_features))
        if i < len(linear_indices) - 1:  # activation after every layer except the last
            layers.append(act_cls())

    model = nn.Sequential(*layers)

    # Map mlp.<idx> -> sequential position. Activations carry no params, so we
    # only need to place the Linear weights; nn.Sequential numbers them the same
    # way (0, 2, 4, ...) given the act-after-linear pattern above.
    remapped = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("mlp."):
            remapped[k[len("mlp."):]] = v  # 'mlp.0.weight' -> '0.weight' to match nn.Sequential
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys when loading actor weights: {missing}")
    return model


def latest_checkpoint(folder: str) -> str:
    """Return the ``model_*.pt`` with the highest iteration number in ``folder``."""
    pattern = os.path.join(folder, "model_*.pt")
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(f"No 'model_*.pt' files found in {folder}")

    def iter_of(path: str) -> int:
        m = re.search(r"model_(\d+)\.pt$", os.path.basename(path))
        return int(m.group(1)) if m else -1

    best = max(candidates, key=iter_of)
    print(f"[INFO] Found {len(candidates)} checkpoint(s) in {folder}; using latest: {os.path.basename(best)}")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Qmini rsl_rl actor checkpoint to ONNX.")
    parser.add_argument("date", help="Date folder under logs/rsl_rl/qmini/; the latest model_*.pt in it is exported")
    parser.add_argument("--activation", default="elu", choices=sorted(ACTIVATIONS), help="Hidden activation (default: elu)")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version (default: 12)")
    parser.add_argument("--key", default="actor_state_dict", help="State-dict key inside the checkpoint")
    args = parser.parse_args()

    checkpoint = latest_checkpoint(os.path.join(DEFAULT_LOG_ROOT, args.date))

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if args.key not in ckpt:
        raise KeyError(f"'{args.key}' not in checkpoint. Available: {list(ckpt.keys())}")
    state_dict = ckpt[args.key]

    model = build_mlp_from_state_dict(state_dict, args.activation).eval()
    obs_dim = model[0].in_features
    act_dim = model[-1].out_features
    print(f"[INFO] Rebuilt actor: obs={obs_dim} -> {act_dim} actions, activation={args.activation}")
    print(model)

    # Always write to the project root as policy.onnx.
    out_path = os.path.join(PROJECT_ROOT, "policy.onnx")

    dummy = torch.zeros(1, obs_dim)
    torch.onnx.export(
        model,
        dummy,
        out_path,
        verbose=False,
        opset_version=args.opset,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
    )
    print(f"[INFO] Wrote ONNX to: {out_path}")

    # Optional parity check against onnxruntime if available.
    try:
        import numpy as np
        import onnxruntime as ort

        sample = torch.rand(1, obs_dim)
        torch_out = model(sample).detach().numpy()
        sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        onnx_out = sess.run(None, {"obs": sample.numpy()})[0]
        max_gap = float(np.abs(torch_out - onnx_out).max())
        print(f"[INFO] PyTorch/ONNX max abs diff: {max_gap:.3e} ({'OK' if max_gap < 1e-5 else 'CHECK'})")
    except ImportError:
        print("[INFO] onnxruntime not installed; skipping parity check. (pip install onnxruntime)")


if __name__ == "__main__":
    main()
