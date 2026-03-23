.PHONY: train-vqbet train-ditflow train-arbet resume check visualize attach lint format test

# Python
export PYTHONPATH := $(CURDIR)/src
PYTHON        := uv run python

# Shared training defaults
DATASET       := lerobot/pusht
DEVICE        := gpu
STEPS         := 200000
BATCH_SIZE    := 64
SAVE_FREQ     := 10000
EVAL_FREQ     := 10000

TRAIN_CMD = $(PYTHON) -m lerobot.scripts.lerobot_train \
	--dataset.repo_id=$(DATASET) \
	--policy.push_to_hub=false \
	--policy.device=$(DEVICE) \
	--steps=$(STEPS) \
	--batch_size=$(BATCH_SIZE) \
	--save_freq=$(SAVE_FREQ) \
	--eval_freq=$(EVAL_FREQ) \
	--wandb.enable=true

# Train VQ-BeT on PushT from scratch
train-vqbet:
	$(TRAIN_CMD) --policy.type=vqbet

# Train DiTFlow on PushT from scratch
train-ditflow:
	$(TRAIN_CMD) --policy.type=ditflow --batch_size=256

# Train ARBeT on PushT from scratch
train-arbet:
	$(TRAIN_CMD) --policy.type=arbet

# Resume any policy from a checkpoint
# Usage: make resume POLICY=vqbet CONFIG_PATH=<path to train_config.json>
resume:
ifndef POLICY
	$(error POLICY is required (vqbet, ditflow, arbet))
endif
ifndef CONFIG_PATH
	$(error CONFIG_PATH is required)
endif
	$(TRAIN_CMD) --policy.type=$(POLICY) \
		--resume=true \
		--config_path=$(CONFIG_PATH)

# Check a saved checkpoint
# Usage: make check CHECKPOINT=<path to checkpoint dir>
check:
ifndef CHECKPOINT
	$(error CHECKPOINT is required. Usage: make check CHECKPOINT=outputs/train/<date>/<time>_vqbet/checkpoints/<step>)
endif
	$(PYTHON) scripts/check_checkpoint.py --checkpoint $(CHECKPOINT)

# Visualize the PushT dataset
visualize:
	$(PYTHON) scripts/visualize_pusht.py

# Lint and format
lint:
	uv run ruff check --fix .

format:
	uv run ruff format .

# Run tests
test:
	uv run pytest tests/ -v

# Attach to a tmux session, creating it if it doesn't exist
# Usage: make attach [SESSION=train]
SESSION := train
attach:
	tmux new-session -A -s $(SESSION)
