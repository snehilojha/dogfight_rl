---
name: verify-change
description: Verification checklist to run before declaring any code change in this repo done or committing — tiered by what the change touches (pure code vs physics/obs/reward vs training pipeline). Use before every commit.
---

# Verifying a change before commit

Match the verification tier to what changed. All commands use
`./.venv/Scripts/python.exe` from the repo root.

## Tier 1 — always (any code change)

```powershell
./.venv/Scripts/python.exe -m pytest -q
```

Full suite, not just the touched test file — geometry/config changes ripple across
env/reward/observation tests. All tests must pass; never skip, xfail, or loosen a
tolerance to get green without user sign-off. If `ruff` is configured (Phase 5+), also
`./.venv/Scripts/python.exe -m ruff check .`.

## Tier 2 — env behavior changed (physics, observation, reward, dogfight_env, rule_based)

Tier 1 plus a smoke train, because unit tests don't catch distribution-level breakage
(NaNs under VecNormalize, action-space mismatches, reward explosions):

```powershell
./.venv/Scripts/python.exe -m training.train --timesteps 2000 --seed 0 --model-out models/smoke --vecnorm-out models/smoke_vecnorm.pkl
```

Check: completes, `ep_rew_mean` finite and plausible, eval callback runs. Delete
`models/smoke*` after.

If the REWARD changed, Tier 2 is not enough — follow the acceptance procedure in the
reward-invariants skill (equal-budget baseline comparison via git worktree).

## Tier 3 — training pipeline changed (train.py, callbacks, self_play, config plumbing)

Tier 2 plus verify the specific artifact contract you touched:

- Checkpointing: confirm both files of the model/vecnorm pair appear and reload
  (`PPO.load(...)` + `VecNormalize.load(...)` round-trip in a snippet).
- Eval/callback changes: confirm the eval env is built from the same `DogfightConfig`
  as training (this exact bug existed once — eval silently ran different physics).
- New CLI flags: run the command with the flag once for real; `--help` is not proof.
- Resume paths (`--resume-from` etc.): actually resume from a real checkpoint once.

## Tier 4 — before a phase is declared complete

Each phase in IMPLEMENTATION_PLAN.md has an explicit "Done when" line and (Phases 2–3)
an acceptance experiment. The phase is not done until that experiment has RUN and its
result is reported to the user — passing tests alone never closes a phase. Long
acceptance runs go through background Bash (training-runs skill).

## Reporting

Report verification honestly: which tiers ran, exact commands, pass/fail with numbers
(test count, ep_rew_mean, win rates). If something was skipped (e.g. acceptance run
pending), say so explicitly instead of implying full verification.
