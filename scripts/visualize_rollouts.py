"""Visualize multiple policy rollouts from the same initial state.

Runs N rollouts of each policy (VQ-BeT, DiTFlow, ARBeT) from the same seed,
collecting agent trajectories. Renders them side by side, overlaid on the
initial Push-T frame, colored by rollout index.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import lerobot_policy_arbet  # noqa: F401  — registers "arbet"
import lerobot_policy_ditflow  # noqa: F401  — registers "ditflow"
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import PushtEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION
from matplotlib.collections import LineCollection

POLICIES = {
    "VQ-BeT": {"path": Path("outputs/train/vqbet_pusht/checkpoints/050000/pretrained_model"), "max_steps": 30},
    "DiTFlow": {"path": Path("outputs/train/ditflow_pusht/checkpoints/050000/pretrained_model"), "max_steps": 40},
    "ARBeT": {"path": Path("outputs/train/arbet_pusht/checkpoints/050000/pretrained_model"), "max_steps": 40},
}

# Initial state: [agent_x, agent_y, block_x, block_y, block_angle]
INITIAL_STATE = [300, 210, 190, 280, 0.785]


def reset_env(env: gym.vector.VectorEnv, seed: int, state: list[float] | None = None):
    """Reset env to a seed or a specific state."""
    if state is not None:
        return env.reset(options={"reset_to_state": state})
    return env.reset(seed=seed)


def run_rollout(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    seed: int,
    max_steps: int,
    state: list[float] | None = None,
) -> list[np.ndarray]:
    """Run one rollout, collecting agent positions."""
    policy.reset()
    observation, _info = reset_env(env, seed, state)

    positions = []
    env_max_steps = env.call("_max_episode_steps")[0]  # type: ignore[attr-defined]
    max_steps = min(max_steps, env_max_steps)
    done = np.array([False])
    step = 0

    while not done[0] and step < max_steps:
        obs = preprocess_observation(observation)
        obs = env_preprocessor(obs)
        obs = preprocessor(obs)

        with torch.inference_mode():
            action = policy.select_action(obs)
        action = postprocessor(action)

        action_transition = env_postprocessor({ACTION: action})
        action_np = action_transition[ACTION].to("cpu").numpy()

        if "agent_pos" in observation:
            positions.append(observation["agent_pos"][0].copy())

        observation, _reward, terminated, truncated, _info = env.step(action_np)
        done = terminated | truncated | done
        step += 1

    if "agent_pos" in observation:
        positions.append(observation["agent_pos"][0].copy())

    return positions


def load_policy_and_processors(model_dir: Path):
    """Load a policy and its pre/post processors."""
    policy_cfg = PreTrainedConfig.from_pretrained(str(model_dir))
    policy_cfg.pretrained_path = model_dir

    # VQ-BeT does not support MPS.
    if str(policy_cfg.device) == "mps" and policy_cfg.type == "vqbet":
        policy_cfg.device = "cpu"

    env_cfg = PushtEnv()
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()

    device_override = {"device_processor": {"device": str(policy_cfg.device)}}
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(model_dir),
        preprocessor_overrides=device_override,
        postprocessor_overrides=device_override,
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg,
        policy_cfg=policy.config,
    )
    return policy, preprocessor, postprocessor, env_preprocessor, env_postprocessor


def main():
    parser = argparse.ArgumentParser(description="Visualize rollouts from VQ-BeT, DiTFlow, and ARBeT.")
    parser.add_argument("--n-rollouts", type=int, default=2)
    parser.add_argument("--output", type=str, default="outputs/rollout_vis.png")
    args = parser.parse_args()

    env_cfg = PushtEnv()
    envs = make_env(cfg=env_cfg, n_envs=1, use_async_envs=False)
    vec_env = envs["pusht"][0]

    # Capture a clean initial frame before any rollouts.
    reset_env(vec_env, seed=0, state=INITIAL_STATE)
    initial_frame = vec_env.call("render")[0]  # type: ignore[attr-defined]
    crop = 10
    cropped = initial_frame[crop:-crop, crop:-crop]
    scale = (initial_frame.shape[0] - 2 * crop) / 512.0

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    cmap = plt.get_cmap("plasma")

    for ax, (name, cfg) in zip(axes, POLICIES.items(), strict=True):
        model_dir = cfg["path"]
        max_steps = cfg["max_steps"]
        assert model_dir.is_dir(), f"Not a directory: {model_dir}"
        print(f"\nLoading {name} from {model_dir}...")
        policy, preprocessor, postprocessor, env_pre, env_post = load_policy_and_processors(model_dir)

        all_positions = []
        for i in range(args.n_rollouts):
            positions = run_rollout(
                vec_env,
                policy,
                env_pre,
                env_post,
                preprocessor,
                postprocessor,
                seed=0,
                max_steps=max_steps,
                state=INITIAL_STATE,
            )
            all_positions.append(np.array(positions))
            print(f"  Rollout {i + 1}: {len(positions)} steps")

        ax.imshow(cropped)
        max_len = max(len(p) for p in all_positions)
        norm = mcolors.Normalize(vmin=0, vmax=max_len - 1)
        for positions in all_positions:
            xs = positions[:, 0] * scale
            ys = positions[:, 1] * scale
            points = np.column_stack([xs, ys]).reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            t = np.arange(len(segments))
            lc = LineCollection(segments, cmap=cmap, norm=norm, alpha=0.7)  # type: ignore[arg-type]
            lc.set_array(t)
            lc.set_linewidth(2)
            ax.add_collection(lc)

        ax.axis("off")
        ax.text(0.5, -0.01, name, transform=ax.transAxes, fontsize=22, ha="center", va="top")

    plt.subplots_adjust(wspace=0.02)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    vec_env.close()
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
