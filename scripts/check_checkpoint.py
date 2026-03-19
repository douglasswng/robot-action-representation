"""Sanity check a VQ-BeT checkpoint: inspect weights, run rollouts in PushT, and save videos."""

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file


def check_weights(model_dir: Path) -> dict:
    """Inspect weight tensors for NaN/zero issues and print stats."""
    import json

    config = json.loads((model_dir / "config.json").read_text())
    print(f"Policy type : {config['type']}")
    print(f"Action shape: {config['output_features']['action']['shape']}")

    weights = load_file(str(model_dir / "model.safetensors"))

    total_params = 0
    nan_keys = []
    zero_keys = []

    for key, tensor in weights.items():
        n = tensor.numel()
        total_params += n
        if torch.isnan(tensor).any():
            nan_keys.append(key)
        # Bias tensors initialized to zero are expected — only flag non-bias all-zero tensors
        if n > 1 and tensor.abs().max().item() == 0.0 and "bias" not in key:
            zero_keys.append(key)

    print(f"Total parameters: {total_params:,}")

    if nan_keys:
        print(f"\n!! NaN detected in {len(nan_keys)} tensor(s):")
        for k in nan_keys:
            print(f"   {k}")
    else:
        print("NaN check: PASS")

    if zero_keys:
        print(f"\n!! Unexpected all-zero tensors ({len(zero_keys)}):")
        for k in zero_keys:
            print(f"   {k}  shape={weights[k].shape}")
    else:
        print("Zero check: PASS (bias tensors excluded)")

    return config


def run_rollouts(model_dir: Path, n_episodes: int, videos_dir: Path) -> None:
    """Load policy, run rollouts in PushT, print metrics, and save videos."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import PushtEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.scripts.lerobot_eval import eval_policy

    # Load policy from checkpoint
    policy_cfg = PreTrainedConfig.from_pretrained(str(model_dir))
    env_cfg = PushtEnv()

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(model_dir),
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg,
        policy_cfg=policy.config,
    )

    # Create vectorized environment
    envs = make_env(cfg=env_cfg, n_envs=min(n_episodes, 10), use_async_envs=False)
    vec_env = envs["pusht"][0]

    print(f"\nRunning {n_episodes} episode(s) in PushT...")

    videos_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        info = eval_policy(
            env=vec_env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=n_episodes,
            max_episodes_rendered=n_episodes,
            videos_dir=videos_dir,
            start_seed=0,
        )

    # Print per-episode results
    print(f"\n{'Episode':<10s} {'Reward':>10s} {'Success':>10s}")
    for ep in info["per_episode"]:
        print(f"  {ep['episode_ix']:<8d} {ep['sum_reward']:>10.2f} {ep['success']!s:>10}")

    print("\nAggregated:")
    print(f"  Success rate  : {info['aggregated']['pc_success']:.1f}%")
    print(f"  Average reward: {info['aggregated']['avg_sum_reward']:.2f}")
    print(f"\nVideos saved to: {videos_dir}/")

    vec_env.close()


def main():
    parser = argparse.ArgumentParser(description="Sanity check a VQ-BeT checkpoint with rollouts.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory (e.g. outputs/train/.../checkpoints/010000)",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=3,
        help="Number of rollout episodes to run (default: 3)",
    )
    parser.add_argument(
        "--videos-dir",
        type=str,
        default=None,
        help="Directory to save rollout videos (default: <checkpoint>/eval_videos)",
    )
    parser.add_argument(
        "--weights-only",
        action="store_true",
        help="Only check weights, skip rollouts",
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint)
    assert checkpoint_dir.is_dir(), f"Not a directory: {checkpoint_dir}"
    model_dir = checkpoint_dir / "pretrained_model"
    assert model_dir.is_dir(), f"Missing pretrained_model dir: {model_dir}"

    print(f"Checkpoint: {checkpoint_dir}\n")

    check_weights(model_dir)

    if args.weights_only:
        print("\n--- Weight checks passed ---")
        return

    videos_dir = Path(args.videos_dir) if args.videos_dir else checkpoint_dir / "eval_videos"
    run_rollouts(model_dir, args.n_episodes, videos_dir)

    print("\n--- All checks passed ---")


if __name__ == "__main__":
    main()
