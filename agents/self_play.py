"""Self-play opponent pool: snapshots of past policies plus scripted baselines.

A snapshot must normalize observations with a *frozen* copy of the VecNormalize
obs statistics captured at snapshot time. Sharing the live training stats would
feed an old policy observations on a different scale than it was trained on,
silently degrading it into a useless opponent.
"""

import copy
import pickle
import shutil
from collections import OrderedDict
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from agents.rule_based import pure_pursuit_policy
from envs.observation import build_obs

# Match VecNormalize's obs-normalization defaults so snapshots see the same
# scaling they were trained under.
CLIP_OBS = 10.0
EPSILON = 1e-8


class SnapshotPolicy:
    """A frozen past policy used as an opponent.

    Normalizes observations with a frozen copy of the VecNormalize obs stats
    (``obs_rms``) captured when the snapshot was taken. ``build_obs`` is
    symmetric in its jet arguments, so the same call produces the observation
    from the opponent's own perspective.
    """

    def __init__(self, model, obs_rms, clip_obs=CLIP_OBS, epsilon=EPSILON):
        self.model = model
        self.obs_rms = obs_rms
        self.clip_obs = clip_obs
        self.epsilon = epsilon

    def _normalize(self, obs):
        if self.obs_rms is None:
            return obs
        return np.clip(
            (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon),
            -self.clip_obs,
            self.clip_obs,
        ).astype(np.float32)

    def __call__(self, own_jet, other_jet, config):
        obs = build_obs(own_jet, other_jet, config)
        action, _ = self.model.predict(self._normalize(obs), deterministic=False)
        return np.asarray(action, dtype=np.float32)


class OpponentPool:
    """Directory-backed pool of policy snapshots plus scripted baselines.

    The pool is fully reconstructable from ``pool_dir`` alone (resume support):
    each snapshot lives in ``pool_dir/gen_{k}/`` as ``model.zip`` +
    ``obs_rms.pkl``. Loaded snapshot models are cached LRU so per-episode
    sampling stays cheap even with a large pool on disk.
    """

    def __init__(
        self,
        pool_dir,
        config,
        max_size=20,
        scripted_prob=0.2,
        scripted_policies=None,
        cache_size=5,
        rng=None,
        clip_obs=CLIP_OBS,
        epsilon=EPSILON,
    ):
        self.pool_dir = Path(pool_dir)
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.max_size = max_size
        self.scripted_prob = scripted_prob
        self.scripted_policies = list(scripted_policies) if scripted_policies else [pure_pursuit_policy]
        self.cache_size = cache_size
        self.rng = rng if rng is not None else np.random.default_rng()
        self.clip_obs = clip_obs
        self.epsilon = epsilon

        self._cache = OrderedDict()  # "gen_{k}" -> SnapshotPolicy
        self._generations = self._discover_generations()

    # --- persistence ---------------------------------------------------------

    def _discover_generations(self):
        gens = []
        for child in self.pool_dir.glob("gen_*"):
            if (child / "model.zip").exists() and (child / "obs_rms.pkl").exists():
                try:
                    gens.append(int(child.name.split("_", 1)[1]))
                except (IndexError, ValueError):
                    continue
        return sorted(gens)

    def add_snapshot(self, model, vecnorm) -> int:
        """Persist a snapshot of ``model`` with a frozen copy of the obs stats.

        Returns the generation index assigned to the snapshot.
        """
        gen = (self._generations[-1] + 1) if self._generations else 0
        gen_dir = self.pool_dir / f"gen_{gen}"
        gen_dir.mkdir(parents=True, exist_ok=True)

        model.save(str(gen_dir / "model.zip"))
        obs_rms = getattr(vecnorm, "obs_rms", None) if vecnorm is not None else None
        with open(gen_dir / "obs_rms.pkl", "wb") as f:
            pickle.dump(copy.deepcopy(obs_rms), f)

        self._generations.append(gen)
        self._evict_if_needed()
        return gen

    def _evict_if_needed(self):
        while len(self._generations) > self.max_size:
            oldest = self._generations.pop(0)
            self._cache.pop(f"gen_{oldest}", None)
            shutil.rmtree(self.pool_dir / f"gen_{oldest}", ignore_errors=True)

    def _load_snapshot(self, gen) -> SnapshotPolicy:
        key = f"gen_{gen}"
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        gen_dir = self.pool_dir / key
        model = PPO.load(str(gen_dir / "model.zip"), device="cpu")
        with open(gen_dir / "obs_rms.pkl", "rb") as f:
            obs_rms = pickle.load(f)

        policy = SnapshotPolicy(model, obs_rms, clip_obs=self.clip_obs, epsilon=self.epsilon)
        self._cache[key] = policy
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return policy

    # --- sampling ------------------------------------------------------------

    def sample(self):
        """Return an opponent policy for the coming episode.

        With probability ``scripted_prob`` (and always when the pool is empty)
        returns a scripted baseline; otherwise a uniformly sampled snapshot.
        """
        if not self._generations or self.rng.random() < self.scripted_prob:
            idx = int(self.rng.integers(len(self.scripted_policies)))
            return self.scripted_policies[idx]
        gen = int(self.rng.choice(self._generations))
        return self._load_snapshot(gen)

    def latest(self):
        """The most recent snapshot policy, or ``None`` if the pool is empty."""
        if not self._generations:
            return None
        return self._load_snapshot(self._generations[-1])

    @property
    def generations(self):
        return list(self._generations)

    def __len__(self):
        return len(self._generations)
