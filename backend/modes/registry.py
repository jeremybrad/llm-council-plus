"""Mode registry - load and list mode definitions from manifests."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModeDefinition:
    """Parsed mode manifest."""

    id: str
    display_name: str
    kind: str  # interactive | analysis | composite
    category: str
    description: str
    version: str
    protocol: dict[str, Any]
    roles: list[dict[str, str]]
    inputs: list[dict[str, str]]
    outputs: list[str]
    stop_criteria: list[str] = field(default_factory=list)
    ui: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, data: dict[str, Any]) -> "ModeDefinition":
        """Create ModeDefinition from parsed YAML data."""
        return cls(
            id=data["id"],
            display_name=data["display_name"],
            kind=data["kind"],
            category=data["category"],
            description=data["description"],
            version=data.get("version", "0.1"),
            protocol=data.get("protocol", {}),
            roles=data.get("roles", []),
            inputs=data.get("inputs", []),
            outputs=data.get("outputs", []),
            stop_criteria=data.get("stop_criteria", []),
            ui=data.get("ui", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "display_name": self.display_name,
            "kind": self.kind,
            "category": self.category,
            "description": self.description,
            "version": self.version,
            "protocol": self.protocol,
            "roles": self.roles,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "stop_criteria": self.stop_criteria,
            "ui": self.ui,
        }


def load_mode(mode_id: str, modes_dir: Path) -> ModeDefinition:
    """Load a mode definition from its manifest file.

    Args:
        mode_id: Mode identifier (folder name)
        modes_dir: Base directory containing mode folders

    Returns:
        ModeDefinition parsed from mode.yaml

    Raises:
        FileNotFoundError: If mode folder or manifest doesn't exist
        ValueError: If manifest is invalid
    """
    mode_path = modes_dir / mode_id
    manifest_path = mode_path / "mode.yaml"

    if not mode_path.exists():
        raise FileNotFoundError(f"Mode not found: {mode_id}")

    if not manifest_path.exists():
        raise FileNotFoundError(f"Mode manifest not found: {manifest_path}")

    with open(manifest_path) as f:
        data = yaml.safe_load(f)

    if not data:
        raise ValueError(f"Empty or invalid manifest: {manifest_path}")

    # Validate required fields
    required = ["id", "display_name", "kind", "category", "description"]
    missing = [f for f in required if f not in data]
    if missing:
        raise ValueError(f"Missing required fields in manifest: {missing}")

    return ModeDefinition.from_yaml(data)


def list_modes(modes_dir: Path) -> list[ModeDefinition]:
    """List all available modes.

    Args:
        modes_dir: Base directory containing mode folders

    Returns:
        List of ModeDefinition objects for valid modes
    """
    modes = []

    if not modes_dir.exists():
        return modes

    for item in modes_dir.iterdir():
        if not item.is_dir():
            continue

        # Skip special directories
        if item.name.startswith("_") or item.name.startswith("."):
            continue

        # Skip if no manifest
        manifest_path = item / "mode.yaml"
        if not manifest_path.exists():
            continue

        try:
            mode = load_mode(item.name, modes_dir)
            modes.append(mode)
        except (ValueError, FileNotFoundError) as e:
            # Log but don't fail on invalid modes
            print(f"Warning: Could not load mode {item.name}: {e}")

    return modes
