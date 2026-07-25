from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from nets import PATCH_SCORE_LAYER_CANDIDATES, WHITEBOX_MODEL_CHOICES


ROUTING_CONFIG_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FrozenRoutingConfig:
    """Globally frozen polarity and architecture-specific routing layers."""

    global_polarity: str
    model_layers: dict[str, str]
    calibration: dict[str, Any]
    schema_version: int = ROUTING_CONFIG_SCHEMA_VERSION
    status: str = "frozen"

    def validate(self) -> None:
        if self.schema_version != ROUTING_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported routing config schema_version={self.schema_version}."
            )
        if self.status != "frozen":
            raise ValueError("routing config status must be 'frozen'.")
        if self.global_polarity not in {"high", "low"}:
            raise ValueError("global_polarity must be high or low.")
        expected_models = set(WHITEBOX_MODEL_CHOICES)
        if set(self.model_layers) != expected_models:
            raise ValueError(
                "model_layers must contain exactly the four registered white-box models."
            )
        for model_name, layer in self.model_layers.items():
            if layer not in PATCH_SCORE_LAYER_CANDIDATES[model_name]:
                raise ValueError(
                    f"invalid frozen layer {layer!r} for {model_name}; choose from "
                    f"{PATCH_SCORE_LAYER_CANDIDATES[model_name]}."
                )
        if not isinstance(self.calibration, dict):
            raise ValueError("calibration metadata must be a JSON object.")

    def layer_for(self, model_name: str) -> str:
        self.validate()
        try:
            return self.model_layers[model_name]
        except KeyError as exc:
            raise ValueError(f"routing config has no layer for {model_name!r}.") from exc

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "global_polarity": self.global_polarity,
            "model_layers": dict(self.model_layers),
            "calibration": self.calibration,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FrozenRoutingConfig":
        config = cls(
            schema_version=int(payload.get("schema_version", -1)),
            status=str(payload.get("status", "")),
            global_polarity=str(payload.get("global_polarity", "")),
            model_layers={
                str(key): str(value)
                for key, value in dict(payload.get("model_layers", {})).items()
            },
            calibration=dict(payload.get("calibration", {})),
        )
        config.validate()
        return config

    @classmethod
    def load(cls, path: Path) -> "FrozenRoutingConfig":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("routing config must contain a JSON object.")
        return cls.from_dict(payload)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
