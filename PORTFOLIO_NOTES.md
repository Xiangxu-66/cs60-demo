# Portfolio Notes

This public demo repository is not a full copy of the private capstone repository.

## Why It Is Trimmed

The original project may contain school, group, dataset, and assessment materials that are not appropriate to publish without explicit permission. This version keeps the engineering portions useful for review while excluding confidential or large artifacts.

## What Reviewers Can Inspect

- Model architecture code and modular pipeline design
- Hydra configuration structure for experiments and ablations
- Training, evaluation, visualization, and reporting scripts
- Unit tests for the core data/model/evaluation paths
- Environment setup for CUDA and Apple Silicon/MPS

## Version Source

This demo was prepared from the local `CS60-1-main` working tree on `main`, based on commit `4a1c2b7` plus the local final configuration adjustments present at export time. It is not a mirror of every private branch from the original team repository.

## Running Locally

Use your own copy of the required image-enhancement dataset and set `DATA_DIR` or the relevant Hydra data override. Generated model weights and experiment outputs should stay local and are ignored by Git.
