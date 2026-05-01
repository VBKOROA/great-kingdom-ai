"""Run a short M6 model inference smoke test.

This script is intended for the Runpod image documented in README:
runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, NoReturn

DEFAULT_CONFIG = Path("configs/m6-smoke.json")


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise ValueError("M6 smoke config must be a JSON object")
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run M6 PyTorch model inference smoke test")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser


def main() -> NoReturn:
    args = build_parser().parse_args()
    config = load_config(args.config)
    device_name = args.device or str(config.get("device", "cuda"))
    require_cuda = bool(config.get("require_cuda", True)) and not args.allow_cpu
    batch_size = int(config.get("batch_size", 8))
    repeat = int(config.get("repeat", 2))
    presets = list(config.get("presets", ["small", "medium"]))

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch is required for M6 smoke test") from exc

    from great_kingdom_ai.features import BOARD_SIZE, FEATURE_CHANNELS
    from great_kingdom_ai.model import create_model

    if require_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA is required by config but is not available")

    device = torch.device(
        device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu"
    )
    for preset in presets:
        model = create_model(str(preset)).to(device)
        model.eval()
        inputs = torch.zeros((batch_size, FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE), device=device)
        with torch.no_grad():
            for _ in range(repeat):
                policy, value = model(inputs)
        print(
            json.dumps(
                {
                    "preset": preset,
                    "device": str(device),
                    "policy_shape": list(policy.shape),
                    "value_shape": list(value.shape),
                },
                sort_keys=True,
            )
        )

    raise SystemExit(0)


if __name__ == "__main__":
    main()
