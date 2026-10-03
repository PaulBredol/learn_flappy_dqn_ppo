#!/usr/bin/env python3
"""V2 PPO agent for Flappy Bird: evaluation capped at 100,000 steps by default.

Basis: train_flappy_ppo_v1.py. Same algorithm, same observations, same rewards.
Changes:
  1. Checkpoints every 50k steps; videos every 250k steps by default.
  2. Evaluation episodes end on a crash or after 100,000 steps by default.
     Videos cover the full evaluation episode without an additional frame cap.
  3. The best video is the single longest flight across ALL checkpoints, not the
     best episode of the checkpoint with the highest mean.
  4. Flight duration in seconds and a printed result table at the end.
Reference: Schulman et al., https://arxiv.org/abs/1707.06347
"""
import argparse
import json
import re
import warnings
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import gymnasium as gym
import numpy as np
import wandb

# Importing the package registers FlappyBird-v0 with Gymnasium.
import flappy_bird_gymnasium  # noqa: F401

from gymnasium.wrappers import PassiveEnvChecker, RecordVideo
from flappy_bird_gymnasium.envs.constants import (
    PLAYER_WIDTH, PLAYER_HEIGHT, PLAYER_MAX_VEL_Y, PIPE_WIDTH, PIPE_HEIGHT,
)
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, CallbackList, CheckpointCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from wandb.integration.sb3 import WandbCallback

CODE_VERSION = "V2_DIRECT_STATE_PPO_EVAL100K"


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
            raise ValueError("PPO V2 expects the original LIDAR environment for unchanged rewards.")
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


class ScoreLoggingCallback(BaseCallback):
    """Log passed pipes per finished training episode, plus the running best.

    SB3 reports only reward and length by itself; the score is the metric that
    matters here, so it goes into TensorBoard and therefore into W&B.
    """

    def __init__(self, window: int = 100):
        super().__init__()
        self.recent_scores: deque[int] = deque(maxlen=window)
        self.best_score = 0
        self.finished_episodes = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", ()):
            # Monitor adds "episode" only once an episode actually ended.
            episode = info.get("episode")
            if episode is None:
                continue
            score = int(episode.get("pipes_passed", info.get("pipes_passed", 0)))
            self.recent_scores.append(score)
            self.best_score = max(self.best_score, score)
            self.finished_episodes += 1
        if self.recent_scores:
            self.logger.record("rollout/score_mean", float(np.mean(self.recent_scores)))
            self.logger.record("rollout/score_recent_max", int(np.max(self.recent_scores)))
            self.logger.record("rollout/score_best", self.best_score)
            self.logger.record("rollout/finished_episodes", self.finished_episodes)
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PPO agent based on Stable-Baselines3 on Flappy Bird."
    )
    parser.add_argument("--env-id", default="FlappyBird-v0")
    parser.add_argument("--total-timesteps", type=int, default=1_500_000)
    parser.add_argument("--checkpoint-freq", type=int, default=150_000,
                        help="Environment steps between checkpoints, summed over all envs.")
    parser.add_argument("--seed", type=int, default=42)

    # PPO collects n_envs * n_steps transitions per update and reuses them for
    # n_epochs passes of minibatch SGD; there is no replay buffer.
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=512,
                        help="Rollout length per environment; batch = n-envs * n-steps.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--final-learning-rate", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--final-clip-range", type=float, default=0.1)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None,
                        help="Optional early stop of the epoch loop; off by default.")
    parser.add_argument("--no-reward-normalization", action="store_true",
                        help="Train on raw rewards instead of VecNormalize running-std rewards.")

    parser.add_argument("--wandb-project", default="flappy-bird-ppo")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--run-name", default=None)

    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument(
        "--device",
        default="cpu",
        help="SB3 device. MLP PPO is fastest on cpu; cuda pays off only for large nets.",
    )
    parser.add_argument("--mode", choices=("train", "eval", "both"), default="both")
    parser.add_argument("--checkpoint-dir", type=Path,
                        help="Existing checkpoint folder; required for --mode eval.")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--video-episodes", type=int, default=1)
    parser.add_argument("--video-freq", type=int, default=250_000)
    parser.add_argument("--video-max-frames", type=int, default=0,
                        help="Frame cap per recorded video; 0 (default) records the full episode.")
    parser.add_argument("--eval-max-steps", type=int, default=100_000,
                        help="Maximum steps per evaluation episode (default: 100000); "
                             "0 explicitly disables the limit.")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"),
                        default="online")
    args = parser.parse_args()
    for name in ("total_timesteps", "checkpoint_freq", "n_envs", "n_steps", "batch_size",
                 "n_epochs", "eval_episodes", "video_freq"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.seed < 0 or args.video_max_frames < 0 or args.eval_max_steps < 0:
        parser.error("seed, video-max-frames and eval-max-steps must be nonnegative")
    if not 0 < args.learning_rate < float("inf"):
        parser.error("learning-rate must be finite and positive")
    if not 0 < args.final_learning_rate <= args.learning_rate:
        parser.error("final-learning-rate must be positive and at most learning-rate")
    if not 0 <= args.gamma <= 1:
        parser.error("gamma must be between 0 and 1")
    if not 0 <= args.gae_lambda <= 1:
        parser.error("gae-lambda must be between 0 and 1")
    if not 0 < args.clip_range < 1:
        parser.error("clip-range must be in (0, 1)")
    if not 0 < args.final_clip_range <= args.clip_range:
        parser.error("final-clip-range must be positive and at most clip-range")
    if args.ent_coef < 0 or args.vf_coef < 0 or args.max_grad_norm <= 0:
        parser.error("ent-coef and vf-coef must be nonnegative, max-grad-norm positive")
    if args.target_kl is not None and args.target_kl <= 0:
        parser.error("target-kl must be positive when given")
    rollout_size = args.n_envs * args.n_steps
    if args.batch_size > rollout_size:
        parser.error("batch-size must not exceed n-envs * n-steps")
    if rollout_size % args.batch_size:
        parser.error(f"n-envs * n-steps ({rollout_size}) must be a multiple of batch-size")
    if not 0 <= args.video_episodes <= args.eval_episodes:
        parser.error("video-episodes must be between 0 and eval-episodes")
    if args.mode != "eval":
        if rollout_size > args.total_timesteps:
            parser.error("total-timesteps must cover at least one rollout")
        if args.checkpoint_freq < rollout_size:
            parser.error("checkpoint-freq must be at least n-envs * n-steps")
        if args.checkpoint_dir is not None:
            parser.error("checkpoint-dir is only used with --mode eval")
    elif args.checkpoint_dir is None or not args.checkpoint_dir.is_dir():
        parser.error("--mode eval requires an existing --checkpoint-dir")
    elif not any(args.checkpoint_dir.glob("*.zip")):
        parser.error("checkpoint-dir contains no model ZIP files")
    return args


def extract_checkpoint_step(checkpoint_path: Path) -> int | None:
    """Extract the timestep from names such as ppo_flappy_bird_50000_steps.zip."""
    match = re.search(r"_(\d+)_steps$", checkpoint_path.stem)
    return int(match.group(1)) if match else None


def checkpoint_sort_key(checkpoint_path: Path) -> tuple[int, int, str]:
    step = extract_checkpoint_step(checkpoint_path)
    if step is None:
        # Non-periodic models, such as final_model.zip, are evaluated last.
        return (1, 0, checkpoint_path.name)
    return (0, step, checkpoint_path.name)


def log_videos(run: Any, key: str, video_dir: Path,
               extra: dict[str, Any] | None = None) -> None:
    """Upload every MP4 in video_dir so the W&B run page shows the flights."""
    if run is None:
        return
    for video_path in sorted(video_dir.glob("*.mp4")):
        payload: dict[str, Any] = {key: wandb.Video(str(video_path), format="mp4")}
        if extra:
            payload.update(extra)
        run.log(payload)


def print_result_table(results: list[dict[str, Any]]) -> None:
    """Print the evaluation overview, so the terminal alone tells the story."""
    print(f"\n{'Checkpoint':>12} {'Mittel':>9} {'bester':>7} {'Laenge':>10} "
          f"{'>=10':>5} {'abgebr.':>8}")
    for result in results:
        print(f"{result['checkpoint_step']:>12} {result['mean_score']:>9.2f} "
              f"{result['best_score']:>7} {result['mean_episode_length']:>10.1f} "
              f"{result['episodes_at_least_ten']:>5} {result['truncated_episodes']:>8}")
    truncated = sum(result["truncated_episodes"] for result in results)
    if truncated:
        print(f"\nACHTUNG: {truncated} Episoden wurden durch --eval-max-steps abgebrochen; "
              f"die Scores dieser Episoden sind Untergrenzen, keine echten Maxima.")
    else:
        print("\nAlle Episoden endeten durch einen echten Absturz - die Scores sind echte Maxima.")


def evaluate_checkpoints(
    checkpoint_dir: Path,
    video_root: Path,
    env_id: str,
    episodes_per_checkpoint: int = 20,
    video_episodes: int = 1,
    max_steps: int = 100_000,
    seed: int = 42,
    device: str = "cpu",
    run: Any = None,
    video_freq: int = 100_000,
    video_max_frames: int = 0,
) -> list[dict[str, Any]]:
    """Evaluate all checkpoints; record periodic videos only at video_freq.

    One additional video of the single longest flight is saved in
    videos/best_model/. The final model is evaluated, but gets no duplicate
    periodic video. max_steps = 0 means no time limit at all.
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
        record_video = bool(video_episodes and step is not None and step > 0
                            and step % video_freq == 0)
        folder_name = f"step_{step}" if step is not None else checkpoint_path.stem
        if folder_name == "final_model":
            folder_name = "last_model"
        checkpoint_video_dir = video_root / folder_name

        model = PPO.load(checkpoint_path, device=device)
        if step is None:
            step = int(model.num_timesteps)
        if model.observation_space.shape != (15,):
            raise ValueError("PPO V2 evaluation requires a 15-input direct-state checkpoint.")

        eval_env = make_flappy_env(
            env_id,
            render_mode="rgb_array" if record_video else None,
            max_episode_steps=max_steps or None,
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
                    video_length=video_max_frames,
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
            "max_episode_length": int(np.max(episode_lengths)),
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
        print(f"{checkpoint_path.name}: Mittel {result['mean_score']:.2f} | "
              f"bester {result['best_score']} | laengste Episode "
              f"{result['max_episode_length']} Schritte")

        # Log checkpoint evaluation summaries into the same W&B run.
        log_data = {
            "eval/checkpoint_step": step,
            "eval/checkpoint_mean_reward": result["mean_reward"],
            "eval/checkpoint_mean_episode_length": result["mean_episode_length"],
            "eval/checkpoint_max_episode_length": result["max_episode_length"],
            "eval/checkpoint_mean_score": result["mean_score"],
            "eval/checkpoint_best_score": result["best_score"],
            "eval/checkpoint_fraction_at_least_ten":
                result["episodes_at_least_ten"] / episodes_per_checkpoint,
        }
        if run is not None:
            run.log(log_data)
            if record_video:
                log_videos(run, "eval/checkpoint_video", checkpoint_video_dir,
                           extra={"eval/checkpoint_step": step})
        # Persist after every model so earlier results survive a later failure.
        (video_root.parent / "evaluation.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )

    print_result_table(results)
    if video_episodes:
        record_best_video(results, checkpoint_dir, video_root, env_id, max_steps,
                          device, run, video_max_frames)
    return results


def select_best_episode(results: list[dict[str, Any]]) -> tuple[dict[str, Any], int]:
    """Pick the single longest flight out of every evaluated episode.

    V1 took the best episode of the checkpoint with the highest mean, which can
    miss a longer flight from a less consistent checkpoint. Ranking is score
    first, then episode length, then the earlier checkpoint.
    """
    candidates = [(result, index)
                  for result in results
                  for index in range(len(result["episode_scores"]))]
    return max(candidates, key=lambda pair: (
        pair[0]["episode_scores"][pair[1]],
        pair[0]["episode_lengths"][pair[1]],
        -pair[0]["checkpoint_step"],
    ))


def record_best_video(results, checkpoint_dir, video_root, env_id, max_steps, device,
                      run=None, video_max_frames=0):
    """Replay the longest evaluated episode of the whole training run."""
    best, episode = select_best_episode(results)
    best_mean = max(results, key=lambda item: item["mean_score"])
    episode_seed = best["episode_seeds"][episode]
    model = PPO.load(checkpoint_dir / best["checkpoint"], device=device)
    destination = video_root / "best_model"
    env = make_flappy_env(env_id, render_mode="rgb_array", max_episode_steps=max_steps or None)
    fps = int(env.metadata.get("render_fps", 30))
    score = 0
    length = 0
    terminated = truncated = False
    try:
        env = RecordVideo(env, video_folder=str(destination),
                          episode_trigger=lambda episode_id: episode_id == 0,
                          name_prefix="best_model", video_length=video_max_frames)
        observation, info = env.reset(seed=episode_seed)
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            score = int(info.get("pipes_passed", score))
            length += 1
    finally:
        env.close()
    info = dict(checkpoint=best["checkpoint"], checkpoint_step=best["checkpoint_step"],
                selection="longest single episode across all checkpoints",
                tie_break="higher score, then longer episode, then earlier checkpoint",
                checkpoint_mean_score=best["mean_score"],
                evaluation_episodes=len(best["episode_scores"]),
                episode_seed=episode_seed, expected_score=best["episode_scores"][episode],
                recorded_score=score, recorded_length=length,
                flight_seconds=round(length / fps, 1),
                video_seconds=round(min(length, video_max_frames or length) / fps, 1),
                truncated=bool(truncated), video_frame_cap=video_max_frames or None,
                episode_step_limit=max_steps or None,
                best_mean_checkpoint=best_mean["checkpoint"],
                best_mean_score=best_mean["mean_score"],
                score_matches_evaluation=score == best["episode_scores"][episode])
    (destination / "selection.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    if not info["score_matches_evaluation"]:
        warnings.warn("Best video replay differs from the evaluated score; see selection.json.")
    if run is not None:
        log_videos(run, "video/best_model", destination)
        run.summary.update({f"best/{key}": value for key, value in info.items()})
    print(f"\nLaengster Flug: {score} Roehren in {length} Schritten "
          f"({info['flight_seconds']} s Spielzeit) | Checkpoint {best['checkpoint']}")
    print(f"Video: {destination}")


def linear_schedule(initial: float, final: float) -> Callable[[float], float]:
    """SB3 schedule: interpolate from initial to final across the whole run."""
    def schedule(progress_remaining: float) -> float:
        completed = min(1.0, max(0.0, 1.0 - progress_remaining))
        return initial + (final - initial) * completed
    return schedule


def create_model(args: argparse.Namespace, train_env: Any, tensorboard_dir: Path) -> PPO:
    return PPO(
        policy="MlpPolicy", env=train_env,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        learning_rate=linear_schedule(args.learning_rate, args.final_learning_rate),
        n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=args.n_epochs,
        gamma=args.gamma, gae_lambda=args.gae_lambda,
        clip_range=linear_schedule(args.clip_range, args.final_clip_range),
        clip_range_vf=None, normalize_advantage=True,
        ent_coef=args.ent_coef, vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm, target_kl=args.target_kl,
        tensorboard_log=str(tensorboard_dir), seed=args.seed,
        device=args.device, verbose=0,
    )


def create_callbacks(args: argparse.Namespace, checkpoint_dir: Path) -> CallbackList:
    # CheckpointCallback counts its own calls, and one call covers n_envs steps.
    callbacks: list[BaseCallback] = [
        ScoreLoggingCallback(),
        CheckpointCallback(
            save_freq=max(1, args.checkpoint_freq // args.n_envs),
            save_path=str(checkpoint_dir), name_prefix="ppo_flappy_bird",
            save_vecnormalize=not args.no_reward_normalization, verbose=0,
        ),
    ]
    if args.wandb_mode != "disabled":
        callbacks.append(WandbCallback(
            gradient_save_freq=0, model_save_freq=0, verbose=0,
        ))
    return CallbackList(callbacks)


def main() -> None:
    args = parse_args()
    print(f"START {CODE_VERSION} | PPO (clipped surrogate + GAE) | "
          f"{args.n_envs} parallele Envs | {args.total_timesteps} Schritte")
    if not args.eval_max_steps:
        print("Auswertung ohne Zeitlimit: eine Episode endet erst, wenn der Vogel stirbt. "
              "Falls eine Episode nicht mehr endet, mit Strg+C abbrechen und "
              "--eval-max-steps setzen.")
    else:
        print(f"Auswertung: maximal {args.eval_max_steps} Schritte pro Episode.")
    if not args.video_max_frames:
        print("Videos ohne Frame-Limit: sie laufen ueber die volle Episode und koennen "
              "entsprechend gross werden.")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    output_dir = Path(args.output_dir) / f"{CODE_VERSION}_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = (args.checkpoint_dir.resolve() if args.mode == "eval"
                      else output_dir / "checkpoints")
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update(run_id=run_id, algorithm="PPO", code_version=CODE_VERSION,
                  base_version="V1_DIRECT_STATE_PPO",
                  compared_against="V10_DIRECT_STATE_DOUBLE_DQN_LR_DECAY",
                  implementation_module="stable_baselines3.PPO",
                  policy_type="on_policy_actor_critic",
                  advantage_estimator="GAE",
                  rollout_size=args.n_envs * args.n_steps,
                  updates=args.total_timesteps // (args.n_envs * args.n_steps),
                  learning_rate_schedule="linear",
                  clip_range_schedule="linear",
                  reward_normalization=not args.no_reward_normalization,
                  observation_normalization=False,
                  episode_step_limit=args.eval_max_steps or None,
                  video_frame_cap=args.video_max_frames or None,
                  best_video_selection="longest single episode across all checkpoints",
                  best_video_directory="videos/best_model",
                  net_arch={"pi": [128, 128], "vf": [128, 128]},
                  observation_mode="direct_simulator_state", normalize_obs=True,
                  environment_use_lidar=True, reward_path="original_lidar",
                  frame_stack=1, observation_dtype="float32", observation_shape=[15],
                  feature_names=FEATURE_NAMES, observation_clip=None,
                  lookahead_pipes=2, includes_offscreen_pipe_state=True,
                  pipe_selection="trailing_edge_ahead_of_bird_left",
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
                name=args.run_name or f"{CODE_VERSION}_{run_id}", config=config,
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
                monitor_kwargs=dict(info_keywords=("pipes_passed",)),
            )
            if not args.no_reward_normalization:
                # Rewards span -10..+10; a running standard deviation keeps the
                # PPO value loss in a workable range. Observations stay raw, so
                # evaluation needs no normalization statistics.
                train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True,
                                         clip_reward=10.0, gamma=args.gamma)
            model = create_model(args, train_env, output_dir / "tensorboard")
            model.learn(
                total_timesteps=args.total_timesteps,
                callback=create_callbacks(args, checkpoint_dir),
                tb_log_name=f"ppo_flappy_{run_id}", progress_bar=True,
            )
            model.save(checkpoint_dir / "final_model")
            if isinstance(train_env, VecNormalize):
                train_env.save(str(checkpoint_dir / "final_vecnormalize.pkl"))

        if args.mode in ("eval", "both"):
            evaluate_checkpoints(
                checkpoint_dir=checkpoint_dir, video_root=output_dir / "videos",
                env_id=args.env_id, episodes_per_checkpoint=args.eval_episodes,
                video_episodes=args.video_episodes, max_steps=args.eval_max_steps,
                seed=args.seed, device=args.device, run=run, video_freq=args.video_freq,
                video_max_frames=args.video_max_frames,
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
