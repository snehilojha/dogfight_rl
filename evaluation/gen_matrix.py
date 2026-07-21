"""Generation-vs-generation win-rate matrix for a self-play pool.

Plays every snapshot generation (as ego) against every other (as opponent) and
reports the win rate. The diagonal is self-play (~0.5 by symmetry); a healthy
curriculum shows the lower-left below 0.5 and the upper-right above it, i.e.
later generations beat earlier ones. The Phase 3 acceptance criterion is
gen N beating gen N-3 more than 60% of the time.

    python -m evaluation.gen_matrix --pool-dir models/pool \
        --config training/hyperparams.yaml --episodes 30 --out gen_matrix.json

Both jets are driven by SnapshotPolicy objects, so each normalizes observations
with its own frozen stats (see agents/self_play.py). Spawns are randomized but
seeded per episode, so every cell of the matrix sees the same set of layouts.
"""

import argparse
import json
from datetime import datetime, timezone

import numpy as np

from agents.self_play import OpponentPool
from envs.config import DogfightConfig
from envs.dogfight_env import DogfightEnv


def _play_episode(ego_policy, opp_policy, config, seed):
    """One episode with ego driven by ego_policy, opponent by opp_policy.

    Returns "win" | "loss" | "timeout" from the ego's perspective.
    """
    env = DogfightEnv(config=config, opponent_policy=opp_policy)
    env.reset(seed=seed)
    terminated = truncated = False
    info = {}
    while not (terminated or truncated):
        action = ego_policy(env.ego_jet, env.opponent_jet, config)
        _, _, terminated, truncated, info = env.step(action)
    env.close()

    events = info.get("events", {})
    if events.get("won", False):
        return "win"
    if events.get("lost", False):
        return "loss"
    return "timeout"


def _win_rate(ego_policy, opp_policy, config, episodes, base_seed):
    wins = 0
    for i in range(episodes):
        if _play_episode(ego_policy, opp_policy, config, base_seed + i) == "win":
            wins += 1
    return wins / max(1, episodes)


def build_matrix(pool_dir, config, episodes, base_seed):
    pool = OpponentPool(pool_dir, config)
    gens = pool.generations
    if not gens:
        raise SystemExit(f"No generations found in {pool_dir}")

    policies = {g: pool._load_snapshot(g) for g in gens}
    matrix = {}
    for i in gens:
        row = {}
        for j in gens:
            if i == j:
                row[j] = None  # diagonal: a policy vs itself, ~0.5 by symmetry
            else:
                row[j] = _win_rate(policies[i], policies[j], config, episodes, base_seed)
        matrix[i] = row
    return gens, matrix


def _acceptance(gens, matrix, gap=3, threshold=0.6):
    """Check gen N beats gen N-gap above threshold; return (passed, checks)."""
    checks = []
    for n in gens:
        earlier = n - gap
        if earlier in gens and matrix[n][earlier] is not None:
            wr = matrix[n][earlier]
            checks.append({"later": n, "earlier": earlier, "win_rate": wr, "pass": wr > threshold})
    passed = bool(checks) and all(c["pass"] for c in checks)
    return passed, checks


def _print_matrix(gens, matrix, checks, passed, gap, threshold):
    print("Generation win-rate matrix (row = ego, col = opponent)\n")
    header = "ego\\opp " + "".join(f"gen_{j:<6}" for j in gens)
    print(header)
    print("-" * len(header))
    for i in gens:
        cells = []
        for j in gens:
            v = matrix[i][j]
            cells.append("  --   " if v is None else f"{v:>5.2f}  ")
        print(f"gen_{i:<4}" + "".join(cells))

    print(f"\nAcceptance: gen N beats gen N-{gap} at >{threshold:.0%}")
    if not checks:
        print(f"  (pool too small — need generations at least {gap} apart)")
    for c in checks:
        mark = "PASS" if c["pass"] else "FAIL"
        print(f"  gen_{c['later']} vs gen_{c['earlier']}: {c['win_rate']:.2f}  [{mark}]")
    print(f"\nResult: {'PASS' if passed else 'FAIL'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", default="models/pool")
    parser.add_argument("--config", default="training/hyperparams.yaml")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--gap", type=int, default=3, help="acceptance compares gen N vs gen N-gap")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--out", default="gen_matrix.json")
    args = parser.parse_args()

    config = DogfightConfig.from_yaml(args.config)
    gens, matrix = build_matrix(args.pool_dir, config, args.episodes, args.base_seed)
    passed, checks = _acceptance(gens, matrix, args.gap, args.threshold)
    _print_matrix(gens, matrix, checks, passed, args.gap, args.threshold)

    report = {
        "metadata": {
            "pool_dir": args.pool_dir,
            "config": args.config,
            "episodes_per_cell": args.episodes,
            "base_seed": args.base_seed,
            "gap": args.gap,
            "threshold": args.threshold,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "generations": gens,
        # JSON keys must be strings.
        "matrix": {str(i): {str(j): matrix[i][j] for j in gens} for i in gens},
        "acceptance": {"passed": passed, "checks": checks},
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
