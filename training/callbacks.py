import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.rule_based import pure_pursuit_policy
from envs.dogfight_env import DogfightEnv


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


def run_eval_episode(model, env):
    obs = env.reset()
    done = False
    total_reward = 0.0
    episode_length = 0
    final_info = None

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = env.step(action)
        total_reward += float(reward[0])
        episode_length += 1
        done = bool(dones[0])
        final_info = infos[0]

    return total_reward, episode_length, final_info


class DogfightEvalCallback(BaseCallback):
    def __init__(self, eval_freq=5000, n_eval_episodes=5, verbose=0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

    def _on_step(self):
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True

        eval_env = DummyVecEnv([lambda: DogfightEnv(opponent_policy=pure_pursuit_policy)])
        if isinstance(self.training_env, VecNormalize):
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
                tmp_path = tmp.name
            self.training_env.save(tmp_path)
            eval_env = VecNormalize.load(tmp_path, eval_env)
            Path(tmp_path).unlink(missing_ok=True)
            eval_env.training = False
            eval_env.norm_reward = False

        wins = 0
        losses = 0
        timeouts = 0
        rewards = []
        lengths = []

        for _ in range(self.n_eval_episodes):
            reward, length, info = run_eval_episode(self.model, eval_env)
            rewards.append(reward)
            lengths.append(length)

            events = info.get("events", {}) if info else {}
            if events.get("won", False):
                wins += 1
            elif events.get("lost", False):
                losses += 1
            else:
                timeouts += 1

        eval_env.close()

        self.logger.record("eval/win_rate", wins / self.n_eval_episodes)
        self.logger.record("eval/loss_rate", losses / self.n_eval_episodes)
        self.logger.record("eval/timeout_rate", timeouts / self.n_eval_episodes)
        self.logger.record("eval/mean_reward", float(np.mean(rewards)))
        self.logger.record("eval/mean_ep_length", float(np.mean(lengths)))
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
