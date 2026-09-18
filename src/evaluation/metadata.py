"""Metadata management for model evaluation outputs.

Handles creation, serialization, and loading of evaluation metadata
including sample indices, file mappings, and model information.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class EvalMetadata:
    """Metadata for a single model evaluation run.

    Attributes:
        model_name: Name/identifier of the model.
        checkpoint_path: Path to the model checkpoint used.
        sampling_strategy: Strategy used to select samples.
        sample_indices: List of test set indices selected.
        total_test_size: Total size of the test set.
        file_mappings: Mapping from test index to output filenames.
        config_snapshot: Optional copy of relevant config sections.
    """

    model_name: str
    checkpoint_path: str
    sampling_strategy: str
    sample_indices: list[int]
    total_test_size: int
    file_mappings: dict[int, dict[str, str]] = field(default_factory=dict)
    config_snapshot: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return asdict(self)

    def save(self, output_path: str | Path) -> None:
        """Save metadata to JSON file.

        Args:
            output_path: Path where metadata JSON will be saved.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "EvalMetadata":
        """Load metadata from JSON file.

        Args:
            path: Path to metadata JSON file.

        Returns:
            EvalMetadata instance.
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        # Convert file_mappings keys back to int
        if "file_mappings" in data:
            data["file_mappings"] = {
                int(k): v for k, v in data["file_mappings"].items()
            }

        return cls(**data)

    def add_file_mapping(
        self,
        test_idx: int,
        input_file: str,
        pred_file: str,
        target_file: str,
    ) -> None:
        """Add file mapping for a single test sample.

        Args:
            test_idx: Original test set index.
            input_file: Path to input image.
            pred_file: Path to prediction image.
            target_file: Path to target image.
        """
        self.file_mappings[test_idx] = {
            "input": input_file,
            "pred": pred_file,
            "target": target_file,
        }


def generate_filename(
    model_name: str,
    file_type: str,
    test_idx: int,
    output_dir: Path | None = None,
) -> str | Path:
    """Generate standardized filename for evaluation outputs.

    Filename format: {type}_{model_name}_{idx:04d}.png

    Args:
        model_name: Name/identifier of the model.
        file_type: Type of file (input, pred, target).
        test_idx: Test set index (0-based).
        output_dir: Optional output directory for full path.

    Returns:
        Filename or full path if output_dir provided.
    """
    if file_type not in ("input", "pred", "target"):
        raise ValueError(f"Invalid file_type: {file_type}")

    # For target files, no model name suffix (shared across models)
    if file_type == "target":
        filename = f"target_{test_idx:04d}.png"
    else:
        filename = f"{file_type}_{model_name}_{test_idx:04d}.png"

    if output_dir:
        return output_dir / filename
    return filename


def extract_model_name(checkpoint_path: str | Path) -> str:
    """Extract model name from checkpoint path.

    Args:
        checkpoint_path: Path to checkpoint file.

    Returns:
        Sanitized model name suitable for filenames.
    """
    path = Path(checkpoint_path)

    # Try to get name from parent directory
    if path.parent.name != "checkpoints":
        name = path.parent.name
    else:
        # Use checkpoint stem
        name = path.stem

    # Clean up common prefixes/suffixes
    for prefix in ("model_", "ckpt_", "epoch_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    for suffix in ("_best", "_final", "_last"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]

    # Remove any remaining non-alphanumeric chars except underscore
    name = "".join(c if c.isalnum() or c == "_" else "_" for c in name)

    return name.lower()


def merge_metadata(
    metadata_paths: list[Path],
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Merge multiple metadata files for comparison.

    Args:
        metadata_paths: List of paths to metadata JSON files.
        output_path: Optional path to save merged metadata.

    Returns:
        Merged metadata dictionary.
    """
    merged = {
        "models": [],
        "common_samples": None,
        "sample_indices": None,
    }

    all_indices = None

    for meta_path in metadata_paths:
        meta = EvalMetadata.load(meta_path)
        model_info = {
            "model_name": meta.model_name,
            "checkpoint_path": meta.checkpoint_path,
            "sampling_strategy": meta.sampling_strategy,
            "file_mappings": meta.file_mappings,
        }
        merged["models"].append(model_info)

        if all_indices is None:
            all_indices = set(meta.sample_indices)
        else:
            all_indices &= set(meta.sample_indices)

    merged["common_samples"] = sorted(all_indices)
    merged["sample_indices"] = merged["common_samples"]

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(merged, f, indent=2)

    return merged
