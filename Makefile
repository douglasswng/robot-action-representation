.PHONY: visualize train-vqbet resume-vqbet check-checkpoint

# Visualize the PushT dataset
visualize:
	uv run python scripts/visualize_pusht.py

# Train VQ-BeT on PushT from scratch
train-vqbet:
	uv run python -m lerobot.scripts.lerobot_train \
		--policy.type=vqbet \
		--dataset.repo_id=lerobot/pusht \
		--policy.push_to_hub=false \
		--policy.device=cpu \
		--steps=100000 \
		--batch_size=8 \
		--save_freq=10000 \
		--eval_freq=10000 \
		--wandb.enable=true

# Resume VQ-BeT training from a checkpoint
# Usage: make resume-vqbet CONFIG_PATH=outputs/train/2025-01-01_12-00-00/pretrained_model/train_config.json
resume-vqbet:
ifndef CONFIG_PATH
	$(error CONFIG_PATH is required. Usage: make resume-vqbet CONFIG_PATH=path/to/train_config.json)
endif
	uv run python -m lerobot.scripts.lerobot_train \
		--policy.type=vqbet \
		--dataset.repo_id=lerobot/pusht \
		--policy.push_to_hub=false \
		--policy.device=cpu \
		--steps=100000 \
		--batch_size=8 \
		--save_freq=10000 \
		--eval_freq=10000 \
		--wandb.enable=true \
		--resume=true \
		--config_path=$(CONFIG_PATH)

# Check a saved checkpoint
# Usage: make check-checkpoint CHECKPOINT=outputs/train/2025-01-01_12-00-00/pretrained_model
check-checkpoint:
ifndef CHECKPOINT
	$(error CHECKPOINT is required. Usage: make check-checkpoint CHECKPOINT=path/to/checkpoint)
endif
	uv run python scripts/check_checkpoint.py --checkpoint $(CHECKPOINT)
