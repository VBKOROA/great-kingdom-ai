"""Versioned KLENT ONNX artifact publication with an atomic current pointer."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from great_kingdom_ai.klent.export import (
    KlentExportKind,
    KlentOnnxParitySummary,
    compare_klent_checkpoint_to_onnx,
    export_klent_checkpoint_to_onnx,
)
from great_kingdom_ai.klent.types import KlentConfig

KLENT_ONNX_MANIFEST_NAME = "manifest.json"
KLENT_ONNX_POINTER_NAME = "current.json"
KLENT_ONNX_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class KlentExportRecord:
    kind: KlentExportKind
    file: str
    output_names: tuple[str, ...]
    parity_passed: bool
    max_policy_abs_diff: float | None
    max_value_abs_diff: float | None
    max_q_values_abs_diff: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "file": self.file,
            "output_names": list(self.output_names),
            "parity_passed": self.parity_passed,
            "max_policy_abs_diff": self.max_policy_abs_diff,
            "max_value_abs_diff": self.max_value_abs_diff,
            "max_q_values_abs_diff": self.max_q_values_abs_diff,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KlentExportRecord:
        return cls(
            kind=str(data["kind"]),  # type: ignore[arg-type]
            file=str(data["file"]),
            output_names=tuple(str(name) for name in data["output_names"]),
            parity_passed=bool(data["parity_passed"]),
            max_policy_abs_diff=_optional_float(data.get("max_policy_abs_diff")),
            max_value_abs_diff=_optional_float(data.get("max_value_abs_diff")),
            max_q_values_abs_diff=_optional_float(data.get("max_q_values_abs_diff")),
        )


@dataclass(frozen=True)
class KlentOnnxManifest:
    model_version: int
    iteration: int
    model_preset: str
    checkpoint_path: str
    klent_config: KlentConfig
    created_at: str
    exports: tuple[KlentExportRecord, ...]
    schema_version: int = KLENT_ONNX_SCHEMA_VERSION
    algorithm: str = "klent"

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "schema_version": self.schema_version,
            "model_version": self.model_version,
            "iteration": self.iteration,
            "model_preset": self.model_preset,
            "checkpoint_path": self.checkpoint_path,
            "klent_config": asdict(self.klent_config),
            "created_at": self.created_at,
            "exports": [record.to_dict() for record in self.exports],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KlentOnnxManifest:
        return cls(
            algorithm=str(data.get("algorithm", "")),
            schema_version=int(data["schema_version"]),
            model_version=int(data["model_version"]),
            iteration=int(data["iteration"]),
            model_preset=str(data["model_preset"]),
            checkpoint_path=str(data["checkpoint_path"]),
            klent_config=KlentConfig(**data["klent_config"]),
            created_at=str(data["created_at"]),
            exports=tuple(
                KlentExportRecord.from_dict(record) for record in data["exports"]
            ),
        )

    def export_for_kind(self, kind: KlentExportKind) -> KlentExportRecord:
        for record in self.exports:
            if record.kind == kind:
                return record
        raise ValueError(f"manifest does not contain a {kind!r} export")


@dataclass(frozen=True)
class KlentOnnxPointer:
    model_version: int
    version_dir: str
    actor_path: str
    eval_path: str
    manifest_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "version_dir": self.version_dir,
            "actor_path": self.actor_path,
            "eval_path": self.eval_path,
            "manifest_path": self.manifest_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KlentOnnxPointer:
        return cls(
            model_version=int(data["model_version"]),
            version_dir=str(data["version_dir"]),
            actor_path=str(data["actor_path"]),
            eval_path=str(data["eval_path"]),
            manifest_path=str(data["manifest_path"]),
        )


def publish_klent_onnx_artifacts(
    checkpoint_path: str | Path,
    work_dir: str | Path,
    *,
    model_version: int,
    iteration: int,
    klent_config: KlentConfig,
    model_preset: str,
    device: str = "cpu",
    precision: str = "fp32",
    check_parity: bool = True,
    overwrite: bool = False,
) -> KlentOnnxManifest:
    """Export eval/actor ONNX into a versioned directory, then publish the pointer.

    The version directory is fully populated (both ONNX files plus manifest)
    before it is renamed into place; the current pointer is replaced last.
    """
    if model_version < 0:
        raise ValueError("model_version must be non-negative")
    if precision == "fp16" and check_parity:
        raise ValueError(
            "CPU parity checks only support fp32 ONNX; export fp16 with check_parity=False"
        )
    root = Path(work_dir) / "onnx"
    version_dir = root / f"version-{model_version:05d}"
    manifest_path = version_dir / KLENT_ONNX_MANIFEST_NAME
    if version_dir.exists() and not overwrite:
        raise FileExistsError(f"KLENT ONNX version already exists: {version_dir}")

    root.mkdir(parents=True, exist_ok=True)
    temporary_dir = root / f".version-{model_version:05d}.tmp"
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir(parents=True)

    records: list[KlentExportRecord] = []
    try:
        export_plan: tuple[tuple[KlentExportKind, str], ...] = (
            ("eval", "eval.onnx"),
            ("actor", "actor.onnx"),
        )
        for kind, file_name in export_plan:
            summary = export_klent_checkpoint_to_onnx(
                checkpoint_path,
                temporary_dir / file_name,
                kind=kind,
                device=device,
                precision=precision,
            )
            parity = (
                compare_klent_checkpoint_to_onnx(
                    checkpoint_path,
                    temporary_dir / file_name,
                    kind=kind,
                )
                if check_parity
                else None
            )
            records.append(_export_record(summary.output_names, kind, file_name, parity))

        failed_parity = [record.kind for record in records if not record.parity_passed]
        if check_parity and failed_parity:
            raise RuntimeError(
                "KLENT ONNX parity check failed; keeping the previous published pointer "
                f"for: {', '.join(failed_parity)}"
            )

        manifest = KlentOnnxManifest(
            model_version=model_version,
            iteration=iteration,
            model_preset=model_preset,
            checkpoint_path=str(checkpoint_path),
            klent_config=klent_config,
            created_at=datetime.now(tz=UTC).isoformat(timespec="seconds"),
            exports=tuple(records),
        )
        (temporary_dir / KLENT_ONNX_MANIFEST_NAME).write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if version_dir.exists():
            shutil.rmtree(version_dir)
        temporary_dir.replace(version_dir)
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)

    published_manifest = load_klent_onnx_manifest(manifest_path)
    pointer = KlentOnnxPointer(
        model_version=model_version,
        version_dir=str(version_dir),
        actor_path=str(version_dir / "actor.onnx"),
        eval_path=str(version_dir / "eval.onnx"),
        manifest_path=str(manifest_path),
    )
    _write_json_atomic(root / KLENT_ONNX_POINTER_NAME, pointer.to_dict())
    return published_manifest


def load_klent_onnx_manifest(path: str | Path) -> KlentOnnxManifest:
    data = _read_json_object(path)
    manifest = KlentOnnxManifest.from_dict(data)
    if manifest.algorithm != "klent":
        raise ValueError(f"manifest algorithm must be 'klent', got {manifest.algorithm!r}")
    if manifest.schema_version != KLENT_ONNX_SCHEMA_VERSION:
        raise ValueError(f"unsupported KLENT ONNX manifest schema: {manifest.schema_version}")
    kinds = {record.kind for record in manifest.exports}
    if kinds != {"eval", "actor"}:
        raise ValueError(f"manifest must contain eval and actor exports, got {sorted(kinds)}")
    return manifest


def load_klent_onnx_pointer(work_dir: str | Path) -> KlentOnnxPointer:
    pointer_path = Path(work_dir) / "onnx" / KLENT_ONNX_POINTER_NAME
    if not pointer_path.exists():
        raise ValueError(f"KLENT ONNX pointer is missing: {pointer_path}")
    return KlentOnnxPointer.from_dict(_read_json_object(pointer_path))


def load_published_klent_onnx(
    work_dir: str | Path,
) -> tuple[KlentOnnxPointer, KlentOnnxManifest]:
    pointer = load_klent_onnx_pointer(work_dir)
    manifest = load_klent_onnx_manifest(pointer.manifest_path)
    if manifest.model_version != pointer.model_version:
        raise ValueError("KLENT ONNX pointer and manifest model versions differ")
    return pointer, manifest


def _export_record(
    output_names: tuple[str, ...],
    kind: KlentExportKind,
    file_name: str,
    parity: KlentOnnxParitySummary | None,
) -> KlentExportRecord:
    return KlentExportRecord(
        kind=kind,
        file=file_name,
        output_names=output_names,
        parity_passed=bool(parity.passed) if parity is not None else False,
        max_policy_abs_diff=None if parity is None else parity.max_policy_abs_diff,
        max_value_abs_diff=None if parity is None else parity.max_value_abs_diff,
        max_q_values_abs_diff=None if parity is None else parity.max_q_values_abs_diff,
    )


def _write_json_atomic(destination: Path, payload: dict[str, Any]) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json_object(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


__all__ = [
    "KLENT_ONNX_MANIFEST_NAME",
    "KLENT_ONNX_POINTER_NAME",
    "KLENT_ONNX_SCHEMA_VERSION",
    "KlentExportRecord",
    "KlentOnnxManifest",
    "KlentOnnxPointer",
    "load_klent_onnx_manifest",
    "load_klent_onnx_pointer",
    "load_published_klent_onnx",
    "publish_klent_onnx_artifacts",
]