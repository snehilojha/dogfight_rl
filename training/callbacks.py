import tempfile
import time
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from agents.rule_based import pure_pursuit_policy
from envs.dogfight_env import DogfightEnv
from evaluation.rollout import run_episode


class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, print_freq=5000, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.print_freq = print_freq
        self._start_time = None

    def _on_training_start(self):
        self._start_time = time.time()

    def _on_step(self):
        if self.n_calls % self.print_freq != 0:
            return True

        elapsed = time.time() - self._start_time
        progress = self.num_timesteps / self.total_timesteps
        if progress > 0:
            eta_secs = elapsed / progress * (1 - progress)
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_secs))
        else:
            eta_str = "--:--:--"

        print(
            f"[{self.num_timesteps:>8}/{self.total_timesteps}]  "
            f"{progress * 100:5.1f}%  elapsed {time.strftime('%H:%M:%S', time.gmtime(elapsed))}  "
            f"ETA {eta_str}"
        )
        return True


def build_frozen_eval_env(training_env, config, opponent_policy):
    """A single-env VecEnv mirroring the training env against a fixed opponent.

    When training uses VecNormalize, the eval env is loaded with a frozen copy
    of the current obs/reward stats (``training=False``, ``norm_reward=False``)
    so evaluation sees the same observation scaling as training without letting
    eval rollouts update the running statistics.
    """
    eval_env = DummyVecEnv([lambda: DogfightEnv(config=config, opponent_policy=opponent_policy)])
    if isinstance(training_env, VecNormalize):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = tmp.name
        training_env.save(tmp_path)
        eval_env = VecNormalize.load(tmp_path, eval_env)
        Path(tmp_path).unlink(missing_ok=True)
        eval_env.training = False
        eval_env.norm_reward = False
    return eval_env


def evaluate_policy_vs(model, training_env, config, opponent_policy, n_episodes):
    """Run ``n_episodes`` of ``model`` vs ``opponent_policy`` and aggregate."""
    eval_env = build_frozen_eval_env(training_env, config, opponent_policy)
    wins = losses = timeouts = 0
    rewards, lengths = [], []

    for _ in range(n_episodes):
        result = run_episode(model, eval_env)
        rewards.append(result.total_reward)
        lengths.append(result.length)
        if result.outcome == "win":
            wins += 1
        elif result.outcome == "loss":
            losses += 1
        else:
            timeouts += 1

    eval_env.close()
    n = max(1, n_episodes)
    return {
        "win_rate": wins / n,
        "loss_rate": losses / n,
        "timeout_rate": timeouts / n,
        "mean_reward": float(np.mean(rewards)),
        "mean_ep_length": float(np.mean(lengths)),
    }


class DogfightEvalCallback(BaseCallback):
    def __init__(self, config=None, eval_freq=5000, n_eval_episodes=5, verbose=0):
        super().__init__(verbose)
        self.config = config
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

    def _on_step(self):
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True

        stats = evaluate_policy_vs(
            self.model, self.training_env, self.config, pure_pursuit_policy, self.n_eval_episodes
        )
        self.logger.record("eval/win_rate", stats["win_rate"])
        self.logger.record("eval/loss_rate", stats["loss_rate"])
        self.logger.record("eval/timeout_rate", stats["timeout_rate"])
        self.logger.record("eval/mean_reward", stats["mean_reward"])
        self.logger.record("eval/mean_ep_length", stats["mean_ep_length"])
        return True


class SelfPlayCallback(BaseCallback):
    """Periodically evaluate the learner and grow the opponent pool.

    Every ``eval_freq`` steps: evaluate vs pure pursuit (a fixed yardstick) and
    vs the latest snapshot, log both to TensorBoard, and add a new snapshot when
    the learner beats the latest one by more than ``snapshot_win_threshold``.
    The first eval seeds generation 0 unconditionally so the pool starts growing
    (until then, ``OpponentPool.sample`` returns scripted opponents, which acts
    as the warm-start phase).
    """

    def __init__(
        self,
        pool,
        config,
        eval_freq=100_000,
        n_eval_episodes=20,
        snapshot_win_threshold=0.55,
        verbose=0,
    ):
        super().__init__(verbose)
        self.pool = pool
        self.config = config
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.snapshot_win_threshold = snapshot_win_threshold

    def _on_step(self):
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True

        pp_stats = evaluate_policy_vs(
            self.model, self.training_env, self.config, pure_pursuit_policy, self.n_eval_episodes
        )
        self.logger.record("selfplay/win_rate_vs_pure_pursuit", pp_stats["win_rate"])
        self.logger.record("selfplay/mean_ep_length", pp_stats["mean_ep_length"])

        latest = self.pool.latest()
        if latest is None:
            gen = self.pool.add_snapshot(self.model, self.training_env)
            if self.verbose:
                print(f"[self-play] seeded snapshot gen {gen}")
        else:
            snap_stats = evaluate_policy_vs(
                self.model, self.training_env, self.config, latest, self.n_eval_episodes
            )
            self.logger.record("selfplay/win_rate_vs_latest", snap_stats["win_rate"])
            if snap_stats["win_rate"] > self.snapshot_win_threshold:
                gen = self.pool.add_snapshot(self.model, self.training_env)
                if self.verbose:
                    print(
                        f"[self-play] added snapshot gen {gen} "
                        f"(win rate vs latest {snap_stats['win_rate']:.2f}, pool size {len(self.pool)})"
                    )

        self.logger.record("selfplay/pool_size", len(self.pool))
        return True


class CheckpointWithVecNormCallback(BaseCallback):
    def __init__(self, save_freq=10000, model_path="models/ppo_dogfight", vecnorm_path="models/vecnormalize.pkl", verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.model_path = model_path
        self.vecnorm_path = vecnorm_path

    def _on_step(self):
        if self.save_freq > 0 and self.n_calls % self.save_freq == 0:
            self.model.save(self.model_path)
            if hasattr(self.training_env, "save"):
                self.training_env.save(self.vecnorm_path)
        return True
