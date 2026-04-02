.PHONY: train-vqbet train-ditflow train-arbet resume attach lint format test

# Python
export PYTHONPATH := $(CURDIR)/src
PYTHON        := uv run python

# Shared training defaults
DATASET       := lerobot/pusht
STEPS         := 200000
BATCH_SIZE    := 64
SAVE_FREQ     := 25000
EVAL_FREQ     := 25000
ENV           := pusht

TRAIN_CMD = $(PYTHON) -m lerobot.scripts.lerobot_train \
	--dataset.repo_id=$(DATASET) \
	--env.type=$(ENV) \
	--policy.push_to_hub=false \
	--steps=$(STEPS) \
	--batch_size=$(BATCH_SIZE) \
	--save_freq=$(SAVE_FREQ) \
	--eval_freq=$(EVAL_FREQ) \
	--wandb.enable=true

# Train VQ-BeT on PushT from scratch
train-vqbet:
	$(TRAIN_CMD) --policy.type=vqbet --steps=250000 --output_dir=outputs/train/vqbet_pusht

# Train DiTFlow on PushT from scratch
train-ditflow:
	$(TRAIN_CMD) --policy.type=ditflow --output_dir=outputs/train/ditflow_pusht

# Train ARBeT on PushT from scratch (uses custom entry point for ScribeTokenDataset wrapping)
train-arbet:
	$(PYTHON) scripts/train_arbet.py \
		--dataset.repo_id=$(DATASET) \
		--env.type=$(ENV) \
		--policy.type=arbet \
		--policy.push_to_hub=false \
		--steps=100000 \
		--batch_size=$(BATCH_SIZE) \
		--save_freq=10000 \
		--eval_freq=10000 \
		--wandb.enable=true \
   		--output_dir=outputs/train/arbet_pusht

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

# Lint and format
lint:
	uv run ruff check --fix .

format:
	uv run ruff format .

# Run tests
test:
	uv run pytest tests/ -v

# Attach to a tmux session, creating it if it doesn't exist
# Usage: make attach [SESSION=vqbet|ditflow|arbet]
SESSION := arbet
attach:
	tmux new-session -A -s $(SESSION)
