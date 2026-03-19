"""Visualize episodes from the Push-T dataset using LeRobot."""

import argparse

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.lerobot_dataset_viz import visualize_dataset


def main():
    parser = argparse.ArgumentParser(description="Visualize Push-T dataset episodes.")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to visualize.")
    args = parser.parse_args()

    dataset = LeRobotDataset("lerobot/pusht")
    print(f"Dataset: {dataset.repo_id}")
    print(f"Number of episodes: {dataset.num_episodes}")
    print(f"Number of frames: {dataset.num_frames}")
    print(f"Features: {list(dataset.features)}")

    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")

    visualize_dataset(dataset=dataset, episode_index=args.episode)


if __name__ == "__main__":
    main()
