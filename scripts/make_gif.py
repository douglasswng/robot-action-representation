"""Produce a side-by-side GIF of VQ-BeT, DiTFlow, and ARBeT on Push-T.

Runs one rollout per policy from the same initial state, captures rendered
frames, stitches them horizontally with labels, and writes an animated GIF.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import lerobot_policy_arbet  # noqa: F401  — registers "arbet"
import lerobot_policy_ditflow  # noqa: F401  — registers "ditflow"
import numpy as np
import torch
from gymnasium.vector import SyncVectorEnv
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import PushtEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION
from PIL import Image, ImageDraw, ImageFont

POLICIES = {
    "VQ-BeT": {"path": Path("outputs/train/vqbet_pusht/checkpoints/050000/pretrained_model"), "max_steps": 300},
    "DiTFlow": {"path": Path("outputs/train/ditflow_pusht/checkpoints/050000/pretrained_model"), "max_steps": 300},
    "ARBeT": {"path": Path("outputs/train/arbet_pusht/checkpoints/050000/pretrained_model"), "max_steps": 300},
}

# Initial state: [agent_x, agent_y, block_x, block_y, block_angle]
INITIAL_STATE = [300, 210, 190, 280, 0.785]

CROP = 10
LABEL_HEIGHT = 36


def reset_env(env: SyncVectorEnv, seed: int, state: list[float] | None = None):
    if state is not None:
        return env.reset(options={"reset_to_state": state})
    return env.reset(seed=seed)


def load_policy_and_processors(model_dir: Path):
    policy_cfg = PreTrainedConfig.from_pretrained(str(model_dir))
    policy_cfg.pretrained_path = model_dir

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


def run_rollout_with_frames(
    env: SyncVectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    seed: int,
    max_steps: int,
    state: list[float] | None = None,
) -> list[np.ndarray]:
    """Run one rollout, collecting rendered frames."""
    policy.reset()
    observation, _info = reset_env(env, seed, state)

    frames = []
    frames.append(env.call("render")[0])

    env_max_steps = env.call("_max_episode_steps")[0]
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

        observation, _reward, terminated, truncated, _info = env.step(action_np)
        done = terminated | truncated | done
        step += 1

        if not done[0]:
            frames.append(env.call("render")[0])

    return frames


def crop_frame(frame: np.ndarray) -> np.ndarray:
    """Crop the grey border from a rendered Push-T frame."""
    return frame[CROP:-CROP, CROP:-CROP]


def add_label(frame: np.ndarray, label: str) -> Image.Image:
    """Crop frame and add a text label below it, matching visualize_rollouts style."""
    cropped = crop_frame(frame)
    h, w = cropped.shape[:2]
    img = Image.new("RGB", (w, h + LABEL_HEIGHT), color=(255, 255, 255))
    img.paste(Image.fromarray(cropped), (0, 0))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    draw.text(((w - text_w) // 2, h + 2), label, fill=(0, 0, 0), font=font)
    return img


def main():
    parser = argparse.ArgumentParser(description="Create a side-by-side GIF of all three policies on Push-T.")
    parser.add_argument("--output", type=str, default="outputs/pusht_comparison.gif")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--max-steps", type=int, default=None, help="Override max steps for all policies")
    args = parser.parse_args()

    env_cfg = PushtEnv()
    envs = make_env(cfg=env_cfg, n_envs=1, use_async_envs=False)
    vec_env = cast(SyncVectorEnv, envs["pusht"][0])

    # Lower the success threshold so episodes end earlier (default is 0.95).
    unwrapped = cast(Any, vec_env.envs[0].unwrapped)
    unwrapped.success_threshold = 0.9

    all_frames: dict[str, list[np.ndarray]] = {}

    for name, cfg in POLICIES.items():
        model_dir = cfg["path"]
        max_steps = args.max_steps if args.max_steps is not None else cfg["max_steps"]
        assert model_dir.is_dir(), f"Not a directory: {model_dir}"
        print(f"Loading {name} from {model_dir}...")
        policy, preprocessor, postprocessor, env_pre, env_post = load_policy_and_processors(model_dir)

        frames = run_rollout_with_frames(
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
        all_frames[name] = frames
        print(f"  {name}: {len(frames)} frames")

    vec_env.close()

    # Pad shorter rollouts by repeating the last frame.
    max_len = max(len(f) for f in all_frames.values())
    for name in all_frames:
        while len(all_frames[name]) < max_len:
            all_frames[name].append(all_frames[name][-1])

    # Stitch frames side by side with labels.
    print(f"Compositing {max_len} frames...")
    names = list(POLICIES.keys())
    composite_frames: list[Image.Image] = []
    for i in range(max_len):
        panels = [add_label(all_frames[n][i], n) for n in names]
        total_w = sum(p.width for p in panels)
        h = panels[0].height
        composite = Image.new("RGB", (total_w, h), color=(255, 255, 255))
        x = 0
        for p in panels:
            composite.paste(p, (x, 0))
            x += p.width
        composite_frames.append(composite)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = 1000 // args.fps
    composite_frames[0].save(
        output_path,
        save_all=True,
        append_images=composite_frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"Saved {len(composite_frames)}-frame GIF to {output_path}")


if __name__ == "__main__":
    main()
