.PHONY: visualize train-vqbet resume-vqbet train-ditflow check-checkpoint attach

# Shared training defaults
DATASET       := lerobot/pusht
DEVICE        := cpu
STEPS         := 100000
BATCH_SIZE    := 8
SAVE_FREQ     := 10000
EVAL_FREQ     := 10000

TRAIN_CMD = uv run python -m lerobot.scripts.lerobot_train \
	--dataset.repo_id=$(DATASET) \
	--policy.push_to_hub=false \
	--policy.device=$(DEVICE) \
	--steps=$(STEPS) \
	--batch_size=$(BATCH_SIZE) \
	--save_freq=$(SAVE_FREQ) \
	--eval_freq=$(EVAL_FREQ) \
	--wandb.enable=true

# Visualize the PushT dataset
visualize:
	uv run python scripts/visualize_pusht.py

# Train VQ-BeT on PushT from scratch
train-vqbet:
	$(TRAIN_CMD) --policy.type=vqbet

# Train DiTFlow on PushT from scratch
train-ditflow:
	$(TRAIN_CMD) --policy.type=ditflow

# Resume VQ-BeT training from a checkpoint
# Usage: make resume-vqbet CONFIG_PATH=outputs/train/2026-03-18/20-05-00_vqbet/checkpoints/050000/pretrained_model/train_config.json
resume-vqbet:
ifndef CONFIG_PATH
	$(error CONFIG_PATH is required. Usage: make resume-vqbet CONFIG_PATH=outputs/train/<date>/<time>_vqbet/checkpoints/<step>/pretrained_model/train_config.json)
endif
	$(TRAIN_CMD) --policy.type=vqbet \
		--resume=true \
		--config_path=$(CONFIG_PATH)

# Check a saved checkpoint
# Usage: make check-checkpoint CHECKPOINT=outputs/train/2026-03-18/20-05-00_vqbet/checkpoints/050000
check-checkpoint:
ifndef CHECKPOINT
	$(error CHECKPOINT is required. Usage: make check-checkpoint CHECKPOINT=outputs/train/<date>/<time>_vqbet/checkpoints/<step>)
endif
	uv run python scripts/check_checkpoint.py --checkpoint $(CHECKPOINT)

# Attach to a tmux session, creating it if it doesn't exist
# Usage: make attach [SESSION=train]
SESSION := train
attach:
	tmux new-session -A -s $(SESSION)
