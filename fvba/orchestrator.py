"""Resumable phase orchestration and run-identity protection."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import tempfile
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import yaml


def experiment_identity(config: Mapping[str, Any]) -> str:
    immutable = deepcopy(dict(config))
    immutable.pop("output", None)
    encoded = json.dumps(immutable, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value):
            raise FloatingPointError("Phase result contains a non-finite NaN value")
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    return value


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        json.dump(_json_safe(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.close()
        os.replace(temporary, path)
    finally:
        if not handle.closed:
            handle.close()
        temporary.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifacts_valid(output_dir: Path, result: Mapping[str, Any]) -> bool:
    artifacts = result.get("artifacts")
    if not artifacts:
        return True
    checksums = result.get("artifact_sha256", {})
    for relative in artifacts:
        path = output_dir / relative
        if not path.is_file():
            return False
        expected = checksums.get(relative)
        if expected is not None and _file_sha256(path) != expected:
            return False
    return True


class RunOrchestrator:
    """Execute named idempotent phases and preserve their completion manifests."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        phases: Mapping[str, Callable[[dict[str, Any]], Mapping[str, Any]]],
    ) -> None:
        self.config = deepcopy(dict(config))
        self.phases = dict(phases)
        self.output_dir = Path(self.config["output"]["dir"])
        self.identity = experiment_identity(self.config)

    def _prepare(self) -> None:
        identity_path = self.output_dir / "experiment_identity.txt"
        if identity_path.exists():
            existing = identity_path.read_text(encoding="utf-8").strip()
            if existing != self.identity:
                raise RuntimeError(
                    "Existing run directory has a different experiment identity; refusing overwrite"
                )
        elif self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise RuntimeError(
                "Existing non-empty run directory has no experiment identity; refusing overwrite"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        identity_path.write_text(self.identity + "\n", encoding="utf-8")
        (self.output_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(self.config, sort_keys=False), encoding="utf-8"
        )
        _write_json_atomic(
            self.output_dir / "environment.json",
            {
                "executable": sys.executable,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "gpu_count": torch.cuda.device_count(),
                "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            },
        )

    def run(self) -> Path:
        self._prepare()
        context: dict[str, Any] = {
            "config": self.config,
            "output_dir": self.output_dir,
            "phase_results": {},
        }
        for name, phase in self.phases.items():
            manifest_path = self.output_dir / "phases" / f"{name}.json"
            force_rebuild = False
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("status") == "complete" and _artifacts_valid(
                    self.output_dir, manifest.get("result", {})
                ):
                    context["phase_results"][name] = manifest.get("result", {})
                    continue
                force_rebuild = True
            context["force_rebuild"] = force_rebuild
            result = dict(phase(context))
            if result.get("artifacts"):
                result["artifact_sha256"] = {
                    relative: _file_sha256(self.output_dir / relative)
                    for relative in result["artifacts"]
                }
            safe_result = _json_safe(result)
            _write_json_atomic(
                manifest_path, {"phase": name, "status": "complete", "result": safe_result}
            )
            context["phase_results"][name] = safe_result
        _write_json_atomic(
            self.output_dir / "summary.json",
            {"experiment_identity": self.identity, "phases": context["phase_results"]},
        )
        return self.output_dir
