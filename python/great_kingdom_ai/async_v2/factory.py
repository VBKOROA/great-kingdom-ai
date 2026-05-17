"""Async v2 factory checkpoint initialization."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from great_kingdom_ai.async_v2.config import FactoryInitV2Config, FactoryInitV2Summary
from great_kingdom_ai.async_v2.paths import _factory_checkpoint_path, _factory_onnx_output_path
from great_kingdom_ai.onnx_export import export_checkpoint_to_onnx
from great_kingdom_ai.pipeline_printer import PipelinePrinter
from great_kingdom_ai.training import TrainingConfig, create_train_state, save_checkpoint


def run_factory_init_v2_once(
    config: FactoryInitV2Config,
    train_config: TrainingConfig,
    *,
    state_factory: Callable[[TrainingConfig], Any] | None = None,
    checkpoint_saver: Callable[[Any, str | Path], Path] | None = None,
    onnx_exporter: Callable[..., Any] | None = None,
    printer: PipelinePrinter | None = None,
) -> FactoryInitV2Summary:
    _validate_factory_init_config(config)
    printer = printer if printer is not None else PipelinePrinter()
    make_state = state_factory if state_factory is not None else create_train_state
    save = checkpoint_saver if checkpoint_saver is not None else save_checkpoint
    export_onnx = onnx_exporter if onnx_exporter is not None else export_checkpoint_to_onnx
    checkpoint_path = _factory_checkpoint_path(config)
    onnx_path = _factory_onnx_output_path(config)
    existing_outputs = [path for path in (checkpoint_path, onnx_path) if path.exists()]
    if existing_outputs and not config.overwrite:
        existing = ", ".join(str(path) for path in existing_outputs)
        raise FileExistsError(f"factory init output already exists: {existing}")
    _require_factory_onnx_export_dependencies(config.onnx_precision)

    printer.title("Async V2 Factory Init")
    printer.metric("work dir", config.work_dir)
    printer.metric("checkpoint", checkpoint_path)
    printer.metric("onnx", onnx_path)
    printer.metric("model preset", train_config.model_preset)
    printer.metric("train device", train_config.device)
    printer.metric("onnx device", config.onnx_device)
    printer.metric("onnx precision", config.onnx_precision)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    printer.step(f"creating factory checkpoint -> {checkpoint_path}")
    state = make_state(train_config)
    saved_checkpoint = Path(save(state, checkpoint_path))
    printer.step(f"exporting factory ONNX -> {onnx_path}")
    temporary_onnx_path = onnx_path.with_suffix(f"{onnx_path.suffix}.tmp")
    export_onnx(
        saved_checkpoint,
        temporary_onnx_path,
        device=config.onnx_device,
        precision=config.onnx_precision,
        dummy_batch_size=config.onnx_dummy_batch_size,
        prefer_ema=True,
    )
    temporary_onnx_path.replace(onnx_path)
    printer.done(f"factory async v2 artifacts ready in {printer.elapsed()}")
    return FactoryInitV2Summary(
        checkpoint_path=saved_checkpoint,
        onnx_output_path=onnx_path,
        model_preset=train_config.model_preset,
        step=int(getattr(state, "step", 0)),
        overwritten=bool(existing_outputs),
    )

def _validate_factory_init_config(config: FactoryInitV2Config) -> None:
    if config.onnx_device not in {"cpu", "cuda"}:
        raise ValueError("onnx_device must be one of: cpu, cuda")
    if config.onnx_precision not in {"fp32", "fp16"}:
        raise ValueError("onnx_precision must be one of: fp32, fp16")
    if config.onnx_dummy_batch_size <= 0:
        raise ValueError("onnx_dummy_batch_size must be positive")

def _require_factory_onnx_export_dependencies(precision: str) -> None:
    missing = []
    for module_name, package_name in (("onnx", "onnx"),):
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            missing.append(package_name)
    if precision == "fp16":
        try:
            importlib.import_module("onnxconverter_common")
        except ModuleNotFoundError:
            missing.append("onnxconverter-common")
    if missing:
        packages = " ".join(missing)
        raise RuntimeError(
            "factory ONNX export dependencies are missing; install them with "
            f"`python -m pip install {packages}` or reinstall the project with "
            "`python -m pip install -e '.[ai]'`"
        )
