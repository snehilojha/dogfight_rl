"""Train the same config across several seeds, sequentially.

Writes each run to ``models/seed_{i}/`` and TensorBoard logs stay under
``runs/``. Deliberately dumb and sequential (CPU-bound PPO; parallel runs would
just starve each other). Feeds the multi-seed analysis notebook.

    python -m training.run_seeds --seeds 0 1 2 3 4 --timesteps 1000000
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/hyperparams.yaml")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--self-play", action="store_true", help="use self_play_train instead of train")
    args = parser.parse_args()

    module = "training.self_play_train" if args.self_play else "training.train"

    for seed in args.seeds:
        out_dir = Path(f"models/seed_{seed}")
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "-m", module,
            "--config", args.config,
            "--seed", str(seed),
            "--model-out", str(out_dir / "model"),
            "--vecnorm-out", str(out_dir / "vecnormalize.pkl"),
        ]
        if args.timesteps is not None:
            cmd += ["--timesteps", str(args.timesteps)]
        if args.self_play:
            cmd += ["--pool-dir", str(out_dir / "pool")]

        print(f"\n=== seed {seed}: {' '.join(cmd)} ===", flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
