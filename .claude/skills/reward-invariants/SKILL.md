---
name: reward-invariants
description: Invariants that must hold whenever envs/reward.py, envs/config.py reward fields, or shaping-related code in dogfight_env.py is modified. Covers potential-based shaping correctness, gamma sync, terminal potential, and the required tests and sanity retrain.
---

# Reward-shaping invariants

The reward uses potential-based shaping (Ng, Harada, Russell 1999), which is the ONLY
form of shaping allowed in this project because it provably does not change the optimal
policy. Any edit to reward code must preserve all of the following.

## The contract

```
phi(s)   = -closing_shaping_scale * toroidal_distance(ego, opp) / arena_diag
F(s, s') = shaping_gamma * phi(s') - phi(s)
```

1. **Shaping is always the exact difference form.** Never add a raw per-step bonus that
   depends on state (distance, speed, angle held over time) — that is exactly the bug
   Phase 2 removed (agents got paid for being chased; timeout episodes farmed shaped
   reward). New dense signals must be expressed as a potential or rejected.
   Exception already grandfathered in: the small `fire_cone_reward` event bonus — do not
   add more like it without an acceptance experiment.
2. **`phi(terminal) = 0`.** On `won`/`lost`, the shaping term uses `next_phi = 0.0`.
   Without this, episode length leaks into return and policy invariance breaks.
3. **`shaping_gamma == PPO gamma`.** `train.py` syncs them
   (`env_config.shaping_gamma = train_config.gamma`); every new training entry point
   must too. If they diverge, shaping is no longer policy-invariant.
4. **The env threads `prev_phi`.** `DogfightEnv.step()` computes
   `prev_phi = potential(ego, opp, config)` BEFORE physics update and passes it in
   `events["prev_phi"]`. `compute_reward` applies shaping only when `prev_phi` is
   present (reset/first-step safety).
5. **Toroidal distance only.** `potential` uses `physics.toroidal_delta`; never
   Euclidean deltas — the arena wraps.

## Required tests (extend, never weaken)

`tests/test_reward.py` — must keep passing, and any new term needs analogous coverage:

- **Telescoping test**: with `shaping_gamma=1.0`, shaping summed over a closed loop of
  states is exactly 0 (`abs_tol=1e-12`). This is the policy-invariance canary — if a
  change breaks this test, the change is wrong, not the test.
- **Terminal-potential test**: reward on a win with `prev_phi` set equals
  `terminal_terms - prev_phi` (i.e. `next_phi` was 0).
- **Direction test**: `closed_in > baseline > fled`.
- Toroidal shortest-path test for `potential` (e.g. x=790 vs x=10 → distance 20).
- Exact-value tests for each event term (win/loss/hit/got_hit/fire-cone).

Run: `./.venv/Scripts/python.exe -m pytest tests/test_reward.py -q`

## Acceptance procedure for any reward change

Never ship a reward change on tests alone:

1. Full suite green.
2. Smoke train: `./.venv/Scripts/python.exe -m training.train --timesteps 2000 --seed 0`
   completes without NaN/explosion (watch `ep_rew_mean` is finite and sane).
3. Baseline comparison at equal budget (see below): same seed, same steps, old vs new
   reward. Acceptance = new is not worse on win rate / episode length. Remember short
   runs (100k) lose to pure pursuit under any reward — compare like against like.

## Baseline-comparison methodology (worktree trick)

To run the OLD code at a past commit while sharing the same venv:

```powershell
git worktree add ../dogfight_baseline <old-commit>
# From inside the worktree, cwd precedence beats the editable install:
cd ../dogfight_baseline
d:\VS_adv_python\dog_fight\.venv\Scripts\python.exe -m training.train --timesteps 100000 --seed 0 --model-out models/baseline --vecnorm-out models/baseline_vecnorm.pkl
```

**Verify the import wins before trusting results**: print `envs.reward.__file__` from the
run's interpreter and grep the loaded source for an old-only symbol. Clean up with
`git worktree remove --force ../dogfight_baseline` afterwards.

## Reward-term logging (Phase 5 prerequisite)

When asked to make rewards debuggable: have `compute_reward` also return/record a
per-term breakdown into `info["reward_terms"]` (dict of term → value). Keep the scalar
return unchanged; tests must assert the terms sum to the scalar.
