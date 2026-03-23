# Robot Action Representation

Comparing diffusion-based, transformer-based, and autoregressive policy architectures for robotic manipulation on the [Push-T](https://github.com/huggingface/gym-pusht) task using the [LeRobot](https://github.com/huggingface/lerobot) framework.

## Policies

| Policy | Type | Description |
|--------|------|-------------|
| **VQ-BeT** | Transformer | Built-in LeRobot VQ-BeT implementation |
| **DiTFlow** | Diffusion | Pi0-inspired flow-based diffusion transformer with velocity ODE formulation and adaptive layer normalization |
| **ARBeT** | Autoregressive | Autoregressive behaviour transformer using [ScribeTokens](https://arxiv.org/abs/2603.02805) — encodes (x,y) actions as BPE-compressed Freeman chain codes and predicts them with cross-entropy loss |

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

### Training

```bash
make train-vqbet     # Train VQ-BeT
make train-ditflow   # Train DiTFlow
make train-arbet     # Train ARBeT
```

Resume from a checkpoint:

```bash
make resume POLICY=<policy> CONFIG_PATH=outputs/train/<date>/<time>_<policy>/checkpoints/<step>/pretrained_model/train_config.json
```

### Evaluation

```bash
make check CHECKPOINT=outputs/train/<date>/<run>/checkpoints/<step>
```

Runs rollouts in the Push-T environment, prints per-episode metrics, and saves videos to `eval_videos/`.

### Visualization

```bash
make visualize   # Visualize Push-T dataset episodes
```

### Testing & Linting

```bash
make test     # Run pytest
make lint     # Lint with ruff (auto-fix)
make format   # Format with ruff
```

## Project Structure

```
├── src/
│   ├── adaln_transformer.py        # Shared AdaLN transformer blocks (DiT/Pi0-style modulation)
│   ├── scribe_tokenizer.py         # (x,y) trajectory ↔ BPE chain-code tokens (wraps tokink)
│   └── lerobot-policy-arbet/       # ARBeT policy package (editable install)
├── vendor/
│   └── lerobot-policy-ditflow/     # DiTFlow policy package (editable install)
├── scripts/
│   ├── check_checkpoint.py         # Checkpoint evaluation + video recording
│   └── visualize_pusht.py          # Dataset visualization
├── tests/                          # Tests for shared modules and ARBeT
├── Makefile                        # Training, eval, and dev commands
└── pyproject.toml
```

## Architecture

All policies share the same data pipeline:

1. **Input**: Push-T dataset — 96x96 RGB images + 2D end-effector state
2. **Vision encoder**: Crop to 84x84, encode with `DiffusionRgbEncoder` (ResNet18 + spatial softmax)
3. **Action prediction**: Predict a chunk of 16 actions, execute the first 8

The policies differ in how they represent and predict actions:

- **DiTFlow** denoises continuous action vectors through 100 Euler ODE steps, conditioned on visual features via AdaLN modulation
- **ARBeT** discretizes (x,y) trajectories into directional tokens (Bresenham decomposition → Freeman chain codes → BPE compression) and predicts them autoregressively

## Key Dependencies

- [LeRobot](https://github.com/huggingface/lerobot) — training framework, dataset loading, environment wrappers
- [gym-pusht](https://github.com/huggingface/gym-pusht) — Push-T simulation
- [tokink](https://pypi.org/project/tokink/) — BPE-compressed chain-code tokenization for digital ink
- [rerun-sdk](https://rerun.io/) — visualization and debugging
