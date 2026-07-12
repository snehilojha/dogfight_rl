# Dogfight AI — Design Notes

A 2D aerial combat reinforcement learning project: two jets fight in a toroidal
800×800 arena. The ego agent is trained with PPO (Stable-Baselines3) against a
scripted pure-pursuit opponent, with self-play planned next (see IMPLEMENTATION_PLAN.md).

All physics, episode, and reward constants live in `envs/config.py`
(`DogfightConfig`) and are overridden via the `env:` section of
`training/hyperparams.yaml`. PPO hyperparameters live in `TrainConfig` /
the `training:` section.

## Action space

```python
gymnasium.spaces.Box(low=[-1, 0, 0], high=[1, 1, 1], shape=(3,), dtype=np.float32)
# action[0]: turn rate  [-1, 1]   (-1=full left, +1=full right)
# action[1]: throttle   [0, 1]    (0=min speed, 1=max speed)
# action[2]: fire       [0, 1]    (>0.5 fires if gun_cooldown==0)
```

## Observation (14-dim, built in `envs/observation.py`)

| Index | Feature | Formula |
|-------|---------|---------|
| 0 | Ego x (normalized) | `x/W * 2 - 1` → `[-1,1]` |
| 1 | Ego y (normalized) | `y/H * 2 - 1` → `[-1,1]` |
| 2 | Ego heading sin | `sin(theta)` |
| 3 | Ego heading cos | `cos(theta)` |
| 4 | Ego speed | `v / max_speed` → `[0,1]` |
| 5 | Relative dx | toroidal `dx / arena_w` |
| 6 | Relative dy | toroidal `dy / arena_h` |
| 7 | Distance | `dist / arena_diag` → `[0,1]` |
| 8 | Bearing sin | `sin(atan2(dy,dx) - ego.theta)` |
| 9 | Bearing cos | `cos(atan2(dy,dx) - ego.theta)` |
| 10 | Opponent heading sin | `sin(opp.theta)` |
| 11 | Opponent heading cos | `cos(opp.theta)` |
| 12 | Gun cooldown | `gun_cooldown / max_cooldown` → `[0,1]` |
| 13 | Health | `health / max_health` → `[0,1]` |

## Key design decisions & rationale

- **PPO over DQN:** Action space is continuous — DQN requires discrete actions,
  producing jerky unrealistic movement.
- **Toroidal arena:** Prevents corner-camping degenerate strategies.
- **Sin/cos angle encoding:** Avoids the 359°→0° gradient discontinuity.
- **VecNormalize mandatory:** Observation features live at different scales;
  normalization stabilizes training.
- **Time penalty per step:** Without it agents learn to survive indefinitely
  (passive strategy wins).
- **Bullet speed 2× jet max speed:** Forces aim lead without making dodging impossible.
- **Self-play over fixed opponent (planned):** A fixed opponent trains a
  specialist; self-play forces generalization.
- **Multi-seed runs (planned):** Single-seed results are statistically weak;
  report mean ± std.

## Checkpoint contract

A usable checkpoint is the model `.zip` **plus** the matching VecNormalize
`.pkl` saved at the same step. Loading one without the other produces incorrect
behaviour. Set `vec_env.training = False` and `norm_reward = False` during eval.
