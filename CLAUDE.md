# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Robotics action representation project for training diffusion-based (DiTFlow), transformer-based (VQ-BeT), and autoregressive (ARBeT) policies on the Push-T manipulation task using the LeRobot framework. Uses UV as the Python package manager.

## Common Commands

### Training
```bash
make train-vqbet          # Train VQ-BeT policy
make train-ditflow        # Train DiTFlow policy
make train-arbet          # Train ARBeT policy
make resume POLICY=<policy> CONFIG_PATH=outputs/train/<date>/<time>_<policy>/checkpoints/<step>/pretrained_model/train_config.json
```

### Testing
```bash
make test                 # Run tests (uv run pytest tests/ -v)
```

Tests cover `adaln_transformer`, `scribe_tokenizer`, `scribe_dataset`, and the ARBeT policy package (configuration, modeling, processor).

### Utilities
```bash
make attach               # Attach to tmux training session
```

### Linting & Formatting
```bash
make lint                 # Lint + auto-fix (ruff check --fix, configured in pyproject.toml, line-length=120, py312)
make format               # Auto-format (ruff format)
```

## Architecture

### Training Pipeline
Training uses `lerobot.scripts.lerobot_train` as the entry point for VQ-BeT and DiTFlow, configured via CLI args in the Makefile. ARBeT uses a custom wrapper (`scripts/train_arbet.py`) that monkey-patches `make_dataset` to inject `ScribeTokenDataset` before calling `lerobot_train.main()`. Outputs go to `outputs/train/<date>/<time>_<policy>/` with checkpoints saved at configurable intervals.

### Policy Implementations
- **VQ-BeT**: Uses the built-in LeRobot `vqbet` policy implementation.
- **DiTFlow**: Custom flow-based diffusion transformer, vendored at `vendor/lerobot-policy-ditflow/` and installed as an editable package. This is a Pi0-inspired architecture using velocity-based ODE formulation with adaptive layer normalization (modulation) in a transformer decoder.
- **ARBeT**: Custom autoregressive behaviour transformer at `src/lerobot-policy-arbet/` installed as an editable package. Uses ScribeTokens (arXiv:2603.02805) to convert integer (x,y) push actions into directional chain-code tokens (8 directions via Bresenham/Freeman), then predicts them autoregressively with cross-entropy loss.

### DiTFlow Architecture (`vendor/lerobot-policy-ditflow/lerobot_policy_ditflow/`)
- `configuration_ditflow.py` — `DiTFlowConfig` dataclass with all hyperparameters
- `modeling_ditflow.py` — `DiTFlowPolicy` (rollout/training entry point), `DiTFlowModel` (encoder + noise net), `_DiTNoiseNet` (transformer decoder with sinusoidal time embedding, Euler ODE solver)
- `processor_ditflow.py` — Pre/post-processing pipelines (normalization, device transfer)

### ARBeT Architecture (`src/lerobot-policy-arbet/lerobot_policy_arbet/`)
- `configuration_arbet.py` — `ARBeTConfig` dataclass: ScribeToken params (BPE vocab size, 8-direction base vocabulary), transformer params (layers, heads, d_model), action chunking (horizon, n_action_steps)
- `modeling_arbet.py` — `ARBeTPolicy` (rollout/training entry point with action queue), `ARBeTModel` (`DiffusionRgbEncoder` vision encoder + causal `AdaLNTransformerDecoder`, cross-entropy loss over token IDs)
- `processor_arbet.py` — Pre/post-processing pipelines (normalization, device transfer)

### Shared Modules (`src/`)
- `adaln_transformer.py` — Policy-agnostic AdaLN transformer building blocks (`AdaLNTransformerDecoder`, `AdaLNDecoderBlock`, `AdaLNFinalLayer`). Provides shift-scale modulation conditioning (DiT/Pi0-style). Used by ARBeT for autoregressive token prediction; DiTFlow could also be refactored to use it.
- `scribe_tokenizer.py` — `ScribeTokenizer` wrapping the `tokink` library (integer (x,y) actions ↔ direction token sequences via Bresenham/Freeman chain codes + BPE, with pen-state tokens stripped and IDs remapped)

Note: The Makefile sets `PYTHONPATH=$(CURDIR)/src` so shared modules are importable directly (e.g., `from adaln_transformer import ...`).

### Data Flow
1. Input: Push-T dataset (96x96 images + 2D state)
2. Vision encoding: Crop to 84x84 → `DiffusionRgbEncoder` (ResNet18 + spatial softmax)
3. DiTFlow predicts action sequence (horizon=16, executes n_action_steps=8); ARBeT uses horizon=32, n_action_steps=2 (n_token_steps=8)
4. DiTFlow uses 100 ODE steps at inference time
5. ARBeT: integer (x,y) actions → Bresenham decomposition → Freeman chain codes (8 directions) → BPE compress → autoregressive token prediction → detokenize to integer coordinates

### Key Dependencies
- **LeRobot** — Policy framework, dataset loading, environment wrappers
- **gym-pusht** — Push-T simulation environment
- **rerun-sdk** — Visualization and debugging
- **tokink** — BPE-compressed chain-code tokenization for digital ink (used by `ScribeTokenizer`)
