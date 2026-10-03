import argparse
import re
import json
import warnings
from hashlib import sha256
from datetime import datetime, timezone
from uuid import uuid4
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import wandb

# Importing the package registers FlappyBird-v0 with Gymnasium.
import flappy_bird_gymnasium  # noqa: F401

from gymnasium.wrappers import RecordVideo
from flappy_bird_gymnasium.envs.constants import (
    PLAYER_WIDTH, PLAYER_HEIGHT, PLAYER_MAX_VEL_Y, PIPE_WIDTH, PIPE_HEIGHT,
)
from gymnasium.wrappers import PassiveEnvChecker
from vector_prioritized_double_dqn import VectorPrioritizedDoubleDQN as DoubleDQN
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from wandb.integration.sb3 import WandbCallback


class FlappyRewardWrapper(gym.Wrapper):
    """Emphasize passing pipes instead of merely staying alive."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.pipes_passed = 0

    def reset(self, **kwargs):
        self.pipes_passed = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, original_reward, terminated, truncated, info = (
            self.env.step(action)
        )

        if np.isclose(original_reward, 1.0):
            reward = 10.0
        elif np.isclose(original_reward, 0.1):
            reward = 0.01
        elif np.isclose(original_reward, -1.0):
            reward = -10.0
        elif np.isclose(original_reward, -0.5):
            reward = 0.0
        else:
            reward = float(original_reward)

        self.pipes_passed = int(info["score"])
        info = dict(info)
        info["pipes_passed"] = self.pipes_passed
        info["original_reward"] = float(original_reward)
        return observation, reward, terminated, truncated, info


FEATURE_NAMES = [
    "bird_center_y", "vertical_velocity", "rotation", "ceiling_clearance", "floor_clearance",
    "pipe1_entry_dx", "pipe1_exit_dx", "pipe1_upper_clearance", "pipe1_lower_clearance", "pipe1_center_dy",
    "pipe2_entry_dx", "pipe2_exit_dx", "pipe2_upper_clearance", "pipe2_lower_clearance", "pipe2_center_dy",
]


class DirectStateObservation(gym.ObservationWrapper):
    """Normalized simulator state, including body margins and two upcoming gaps.

    Positive clearance means the body fits on that side. Positive center_dy
    means the opening center is below the bird. Distances stay signed; we do
    not clip away overlaps or states above the screen. Simulator pipe data
    includes the upcoming opening even before it enters the visible screen.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if env.observation_space.shape != (180,):
            raise ValueError("V12 expects the original LIDAR environment for unchanged rewards.")
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (15,), dtype=np.float32)

    def observation(self, observation):
        state = self.env.unwrapped
        width, height = float(state._screen_width), float(state._screen_height)
        left, top = float(state._player_x), float(state._player_y)
        right, bottom = left + PLAYER_WIDTH, top + PLAYER_HEIGHT
        center = top + PLAYER_HEIGHT / 2
        # A pipe remains relevant until its trailing edge clears the entire
        # bird, NOT merely until the environment awards the score at mid-pipe.
        pipes = sorted(
            [(up, low) for up, low in zip(state._upper_pipes, state._lower_pipes)
             if float(up["x"]) + PIPE_WIDTH > left],
            key=lambda pair: pair[0]["x"],
        )
        if len(pipes) < 2:
            raise ValueError("Expected at least two not-fully-passed pipes.")
        values = [center / height, state._player_vel_y / PLAYER_MAX_VEL_Y,
                  state._player_rot / 90.0, top / height,
                  (state._ground["y"] - 1 - bottom) / height]
        for up, low in pipes[:2]:
            gap_top = float(up["y"]) + PIPE_HEIGHT
            gap_bottom = float(low["y"])
            values.extend([
                (float(up["x"]) - right) / width,
                (float(up["x"]) + PIPE_WIDTH - left) / width,
                (top - gap_top) / height,
                (gap_bottom - bottom) / height,
                ((gap_top + gap_bottom) / 2 - center) / height,
            ])
        result = np.asarray(values, dtype=np.float32)
        if result.shape != (15,) or not np.isfinite(result).all():
            raise ValueError("Invalid direct state: expected 15 finite values.")
        return result


def make_flappy_env(env_id: str, **kwargs) -> gym.Env:
    """Change only the observations; retain the original LIDAR reward path."""
    env = gym.make(env_id, use_lidar=True, normalize_obs=True,
                   disable_env_checker=True, **kwargs)
    env = PassiveEnvChecker(DirectStateObservation(env))
    return FlappyRewardWrapper(env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Double DQN agent based on Stable-Baselines3 on Flappy Bird."
    )
    parser.add_argument("--env-id", default="FlappyBird-v0")
    parser.add_argument("--total-timesteps", type=int, default=1_500_000)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--final-learning-rate", type=float, default=1e-5)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--no-reward-normalization", action="store_true")
    parser.add_argument("--per-alpha", type=float, default=0.6)
    parser.add_argument("--per-beta-start", type=float, default=0.4)
    parser.add_argument("--per-epsilon", type=float, default=1e-6)
    parser.add_argument("--buffer-size", type=int, default=200_000)  # Stored transitions, not episodes.
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--exploration-fraction", type=float, default=1 / 3)
    parser.add_argument("--exploration-final-eps", type=float, default=0.02)

    parser.add_argument("--wandb-project", default="flappy-bird-dqn")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--run-name", default=None)

    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument(
        "--device",
        default="auto",
        help="SB3 device, such as auto, cpu, cuda, or cuda:0.",
    )
    parser.add_argument("--mode", choices=("train", "eval", "both"), default="both")
    parser.add_argument("--checkpoint-dir", type=Path,
                        help="Existing checkpoint folder; required for --mode eval.")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--video-episodes", type=int, default=1)
    parser.add_argument("--video-freq", type=int, default=150_000)
    parser.add_argument("--eval-max-steps", type=int, default=100_000,
                        help="Maximum steps per evaluation episode.")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"),
                        default="online")
    args = parser.parse_args()
    for name in ("total_timesteps", "checkpoint_freq", "buffer_size", "batch_size",
                 "eval_episodes", "eval_max_steps", "video_freq", "n_envs"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 < args.final_learning_rate <= args.learning_rate:
        parser.error("final-learning-rate must be positive and at most learning-rate")
    if args.buffer_size < args.n_envs or args.buffer_size % args.n_envs:
        parser.error("buffer-size must be a positive multiple of n-envs")
    if args.checkpoint_freq % args.n_envs or 1000 % args.n_envs:
        parser.error("n-envs must divide checkpoint-freq and target update interval 1000")
    if args.seed < 0 or args.learning_starts < 0:
        parser.error("seed and learning-starts must be nonnegative")
    if not 0 < args.learning_rate < float("inf"):
        parser.error("learning-rate must be finite and positive")
    if not 0 <= args.gamma <= 1:
        parser.error("gamma must be between 0 and 1")
    if not 0 < args.exploration_fraction <= 1:
        parser.error("exploration-fraction must be in (0, 1]")
    if not 0 <= args.exploration_final_eps <= 1:
        parser.error("exploration-final-eps must be between 0 and 1")
    if not 0 <= args.video_episodes <= args.eval_episodes:
        parser.error("video-episodes must be between 0 and eval-episodes")
    if args.batch_size > args.buffer_size:
        parser.error("batch-size must not exceed buffer-size")
    if args.mode != "eval":
        if args.learning_starts >= args.total_timesteps:
            parser.error("learning-starts must be smaller than total-timesteps")
        if args.total_timesteps * args.exploration_fraction <= args.learning_starts:
            parser.error("Exploration decay must end after learning-starts; increase "
                         "total-timesteps or exploration-fraction")
        if args.checkpoint_dir is not None:
            parser.error("checkpoint-dir is only used with --mode eval")
    elif args.checkpoint_dir is None or not args.checkpoint_dir.is_dir():
        parser.error("--mode eval requires an existing --checkpoint-dir")
    elif not any(args.checkpoint_dir.glob("*.zip")):
        parser.error("checkpoint-dir contains no model ZIP files")
    if not 0 <= args.per_alpha <= 1 or not 0 <= args.per_beta_start <= 1:
        parser.error("PER alpha and beta must be in [0,1]")
    if not 0 < args.per_epsilon < float("inf"):
        parser.error("PER epsilon must be finite and positive")
    return args

def extract_checkpoint_step(checkpoint_path: Path) -> int | None:
    """Extract the timestep from names such as dqn_flappy_50000_steps.zip."""
    match = re.search(r"_(\d+)_steps$", checkpoint_path.stem)
    return int(match.group(1)) if match else None


def checkpoint_sort_key(checkpoint_path: Path) -> tuple[int, int, str]:
    step = extract_checkpoint_step(checkpoint_path)
    if step is None:
        # Non-periodic models, such as final_model.zip, are evaluated last.
        return (1, 0, checkpoint_path.name)
    return (0, step, checkpoint_path.name)

def evaluate_checkpoints(
    checkpoint_dir: Path,
    video_root: Path,
    env_id: str,
    episodes_per_checkpoint: int = 20,
    video_episodes: int = 2,
    max_steps: int = 100_000,
    seed: int = 42,
    device: str = "auto",
    run: Any = None,
    video_freq: int = 150_000,
) -> list[dict[str, Any]]:
    """Evaluate all checkpoints; record periodic videos only at video_freq.

    One additional best-episode video is saved in videos/best_model/.
    The final model is evaluated, but gets no duplicate periodic video.
    """
    checkpoint_paths = sorted(
        checkpoint_dir.glob("*.zip"),
        key=checkpoint_sort_key,
    )

    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No .zip model checkpoints found in {checkpoint_dir.resolve()}"
        )

    video_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    for checkpoint_path in checkpoint_paths:
        step = extract_checkpoint_step(checkpoint_path)
        record_video = bool(video_episodes and step is not None and step > 0 and step % video_freq == 0)
        folder_name = f"step_{step}" if step is not None else checkpoint_path.stem
        if folder_name == "final_model":
            folder_name = "last_model"
        checkpoint_video_dir = video_root / folder_name

        model = DoubleDQN.load(checkpoint_path, device=device)
        if step is None:
            step = int(model.num_timesteps)
        if model.observation_space.shape != (15,):
            raise ValueError("V12 evaluation requires a 15-input direct-state checkpoint; "
                             "evaluate V3/V4/V5 with their original scripts.")

        eval_env = make_flappy_env(
            env_id,
            render_mode="rgb_array" if record_video else None,
            max_episode_steps=max_steps,
        )

        episode_rewards: list[float] = []
        episode_lengths: list[int] = []
        episode_scores: list[int] = []

        truncated_episodes = 0
        try:
            if record_video:
                eval_env = RecordVideo(
                    env=eval_env,
                    video_folder=str(checkpoint_video_dir),
                    episode_trigger=lambda episode_id: episode_id < video_episodes,
                    video_length=0,
                    name_prefix=checkpoint_path.stem,
                )
            for episode_index in range(episodes_per_checkpoint):
                observation, info = eval_env.reset(
                    seed=seed + episode_index
                )
                terminated = False
                truncated = False
                episode_reward = 0.0
                episode_length = 0
                pipes_passed = 0

                while not (terminated or truncated):
                    action, _ = model.predict(
                        observation,
                        deterministic=True,
                    )
                    observation, reward, terminated, truncated, info = (
                        eval_env.step(action)
                    )
                    episode_reward += float(reward)
                    episode_length += 1
                    pipes_passed = int(info.get("pipes_passed", pipes_passed))

                truncated_episodes += int(truncated)
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episode_scores.append(pipes_passed)
        finally:
            # Closing flushes and finalizes the MP4 files.
            eval_env.close()

        result = {
            "checkpoint": checkpoint_path.name,
            "checkpoint_step": step,
            "mean_reward": float(np.mean(episode_rewards)),
            "reward_std": float(np.std(episode_rewards)),
            "truncated_episodes": truncated_episodes,
            "mean_episode_length": float(np.mean(episode_lengths)),
            "mean_score": float(np.mean(episode_scores)),
            "best_score": int(np.max(episode_scores)),
            "episodes_at_least_two": sum(score >= 2 for score in episode_scores),
            "episodes_at_least_ten": sum(score >= 10 for score in episode_scores),
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "episode_scores": episode_scores,
            "episode_seeds": list(range(seed, seed + episodes_per_checkpoint)),
            "video_directory": str(checkpoint_video_dir) if record_video else None,
        }
        results.append(result)

        # Log checkpoint evaluation summaries into the same W&B run.
        log_data = {
            "eval/checkpoint_mean_reward": result["mean_reward"],
            "eval/checkpoint_mean_episode_length": result[
                "mean_episode_length"
            ],
            "eval/checkpoint_mean_score": result["mean_score"],
            "eval/checkpoint_best_score": result["best_score"],
            "eval/checkpoint_fraction_at_least_ten": result["episodes_at_least_ten"] / episodes_per_checkpoint,
        }
        if step is not None:
            log_data["eval/checkpoint_step"] = step
        if run is not None:
            run.log(log_data)
        # Persist after every model so earlier results survive a later failure.
        (video_root.parent / "evaluation.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )

    if video_episodes:
        record_best_video(results, checkpoint_dir, video_root, env_id, max_steps, device)
    return results


def select_best_episode(results: list[dict[str, Any]]) -> tuple[dict[str, Any], int]:
    """Rank checkpoints by mean score; ties keep the first evaluated checkpoint."""
    best = max(results, key=lambda item: item["mean_score"])
    episode = max(range(len(best["episode_scores"])),
                  key=lambda index: best["episode_scores"][index])
    return best, episode


def record_best_video(results, checkpoint_dir, video_root, env_id, max_steps, device):
    """Replay the best evaluated episode of the checkpoint with the highest mean."""
    best, episode = select_best_episode(results)
    episode_seed = best["episode_seeds"][episode]
    model = DoubleDQN.load(checkpoint_dir / best["checkpoint"], device=device)
    destination = video_root / "best_model"
    env = make_flappy_env(env_id, render_mode="rgb_array", max_episode_steps=max_steps)
    score = 0
    length = 0
    terminated = truncated = False
    try:
        env = RecordVideo(env, video_folder=str(destination),
                          episode_trigger=lambda episode_id: episode_id == 0,
                          name_prefix="best_model", video_length=0)
        observation, info = env.reset(seed=episode_seed)
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            score = int(info.get("pipes_passed", score))
            length += 1
    finally:
        env.close()
    info = dict(checkpoint=best["checkpoint"], checkpoint_step=best["checkpoint_step"],
                selection="highest mean_score; best episode of that checkpoint",
                tie_break="first evaluated checkpoint / first episode",
                mean_score=best["mean_score"], evaluation_episodes=len(best["episode_scores"]),
                episode_seed=episode_seed, expected_score=best["episode_scores"][episode],
                recorded_score=score, recorded_length=length, truncated=bool(truncated),
                score_matches_evaluation=score == best["episode_scores"][episode])
    (destination / "selection.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    if not info["score_matches_evaluation"]:
        warnings.warn("Best video replay differs from the evaluated score; see selection.json.")
    print(f"Best video: {destination} | checkpoint: {best['checkpoint']} | score: {score}")


def linear_learning_rate(initial, final):
    def schedule(progress_remaining):
        progress = min(1.0, max(0.0, progress_remaining))
        return final + (initial - final) * progress
    return schedule


def create_model(args: argparse.Namespace, train_env: Any, tensorboard_dir: Path) -> DoubleDQN:
    return DoubleDQN(
        policy="MlpPolicy", env=train_env,
        per_beta_start=args.per_beta_start,
        replay_buffer_kwargs=dict(alpha=args.per_alpha, priority_epsilon=args.per_epsilon),
        policy_kwargs=dict(net_arch=[128, 128]),
        learning_rate=linear_learning_rate(args.learning_rate, args.final_learning_rate), buffer_size=args.buffer_size,
        learning_starts=args.learning_starts, batch_size=args.batch_size,
        gamma=args.gamma, train_freq=4, gradient_steps=args.n_envs,
        target_update_interval=1_000,
        exploration_fraction=args.exploration_fraction,
        exploration_initial_eps=1.0, exploration_final_eps=args.exploration_final_eps,
        tensorboard_log=str(tensorboard_dir), seed=args.seed,
        device=args.device, verbose=0,
    )


def create_callbacks(args: argparse.Namespace, checkpoint_dir: Path) -> CallbackList:
    callbacks = [CheckpointCallback(
        save_freq=args.checkpoint_freq // args.n_envs, save_path=str(checkpoint_dir),
        name_prefix="dqn_flappy_bird", save_replay_buffer=False,
        save_vecnormalize=not args.no_reward_normalization, verbose=0,
    )]
    if args.wandb_mode != "disabled":
        callbacks.append(WandbCallback(
            gradient_save_freq=0, model_save_freq=0, verbose=0,
        ))
    return CallbackList(callbacks)


def main() -> None:
    args = parse_args()
    print(f"START V12 | Double DQN + PER | {args.n_envs} Envs | LR {args.learning_rate} -> {args.final_learning_rate}")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    output_dir = Path(args.output_dir) / f"V12_DIRECT_STATE_DOUBLE_DQN_PER_PPO_STYLE_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = (args.checkpoint_dir.resolve() if args.mode == "eval"
                      else output_dir / "checkpoints")
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update(run_id=run_id, algorithm="DoubleDQN_PER", base_version="V11_DIRECT_STATE_DOUBLE_DQN_PER",
                  code_version="V12_DIRECT_STATE_DOUBLE_DQN_PER_PPO_STYLE",
                  target_action_selection="online_network_argmax",
                  target_action_evaluation="target_network_gather",
                  implementation_module="vector_prioritized_double_dqn.py",
                  implementation_sha256=sha256(
                      Path(__file__).with_name("vector_prioritized_double_dqn.py").read_bytes()).hexdigest(),
                  parent_per_sha256=sha256(Path(__file__).with_name("prioritized_double_dqn.py").read_bytes()).hexdigest(),
                  learning_rate_schedule="linear_over_training",
                  reward_normalization=not args.no_reward_normalization,
                  reward_clip=10.0, observation_normalization=False,
                  vector_backend="DummyVecEnv", n_steps=1,
                  updates_per_transition=0.25,
                  per_beta_end=1.0, per_beta_schedule="linear_over_training",
                  per_replacement="FIFO", per_weight_normalization="buffer_global_max",
                  double_dqn_sha256=sha256(Path(__file__).with_name("double_dqn.py").read_bytes()).hexdigest(),
                  exploration_decay_steps=args.total_timesteps * args.exploration_fraction,
                  best_video_selection="highest checkpoint mean_score, then highest episode score",
                  best_video_directory="videos/best_model",
                  net_arch=[128, 128],
                  observation_mode="direct_simulator_state", normalize_obs=True,
                  environment_use_lidar=True, reward_path="original_lidar",
                  frame_stack=1, observation_dtype="float32", observation_shape=[15],
                  feature_names=FEATURE_NAMES, observation_clip=None,
                  lookahead_pipes=2, includes_offscreen_pipe_state=True,
                  pipe_selection="trailing_edge_ahead_of_bird_left",
                  train_freq=4, gradient_steps=args.n_envs, target_update_interval=1000,
                  checkpoint_replay_buffer=False, final_replay_buffer=True,
                  rewards={"pipe": 10.0, "alive": 0.01,
                           "death": -10.0, "proximity": 0.0})
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    run = None
    train_env = None
    succeeded = False
    try:
        if args.wandb_mode != "disabled":
            run = wandb.init(
                project=args.wandb_project, entity=args.wandb_entity,
                name=args.run_name or f"V12_DIRECT_STATE_DOUBLE_DQN_PER_PPO_STYLE_{run_id}", config=config,
                mode=args.wandb_mode, dir=str(output_dir),
                sync_tensorboard=True, save_code=True,
                settings=wandb.Settings(console="off"),
            )
            run.define_metric("eval/checkpoint_step")
            run.define_metric("eval/checkpoint_*", step_metric="eval/checkpoint_step")
        if args.mode in ("train", "both"):
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            train_env = make_vec_env(
                env_id=lambda: make_flappy_env(args.env_id),
                n_envs=args.n_envs, seed=args.seed,
                monitor_dir=str(output_dir / "monitor"),
            )
            if not args.no_reward_normalization:
                train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True,
                                         clip_reward=10.0, gamma=args.gamma)
            model = create_model(args, train_env, output_dir / "tensorboard")
            model.learn(
                total_timesteps=args.total_timesteps,
                callback=create_callbacks(args, checkpoint_dir),
                tb_log_name=f"dqn_flappy_{run_id}", progress_bar=True,
            )
            model.save(checkpoint_dir / "final_model")
            model.save_replay_buffer(checkpoint_dir / "final_replay_buffer.pkl")
            if isinstance(train_env, VecNormalize):
                train_env.save(str(checkpoint_dir / "final_vecnormalize.pkl"))

        if args.mode in ("eval", "both"):
            evaluate_checkpoints(
                checkpoint_dir=checkpoint_dir, video_root=output_dir / "videos",
                env_id=args.env_id, episodes_per_checkpoint=args.eval_episodes,
                video_episodes=args.video_episodes, max_steps=args.eval_max_steps,
                seed=args.seed, device=args.device, run=run, video_freq=args.video_freq,
            )
        succeeded = True
    finally:
        # Attempt both cleanup operations, even if one fails. Preserve original errors.
        if train_env is not None:
            try:
                train_env.close()
            except Exception as error:
                warnings.warn(f"Could not close training environment: {error}")
        if run is not None:
            try:
                run.finish(exit_code=0 if succeeded else 1)
            except Exception as error:
                warnings.warn(f"Could not finish W&B run: {error}")

if __name__ == "__main__":
    main()
