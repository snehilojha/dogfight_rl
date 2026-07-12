# Dogfight + COLOSSEUM Roadmap

This document replaces the split between the original dogfight build plan and the larger RL stack vision.

The goal is to keep the project focused on two things:

1. Build a strong 2D dogfight RL project that actually works end to end.
2. Grow that project into COLOSSEUM in stages, without losing the full long-term vision.

This roadmap is aligned to the current repo state in `d:\VS_adv_python\dog_fight`.

## Guiding Principle

Dogfight AI is the product.

COLOSSEUM is the expansion layer that turns one trained agent into a competitive training ecosystem.

That means the base simulator, environment, reward, training loop, checkpointing, and evaluation come first. COLOSSEUM only starts after the base dogfight project is stable enough to produce believable fighters.

## Current Repo State

The repo already has a partial foundation, but most of the stack is still unbuilt.

Implemented or partially implemented:

- `envs/physics.py`: mostly present
- `envs/observation.py`: present and broadly aligned with the design
- `training/hyperparams.yaml`: minimal config stub

Still empty or nearly empty:

- `envs/dogfight_env.py`
- `envs/reward.py`
- `agents/ppo_agent.py`
- `agents/rule_based.py`
- `agents/self_play.py`
- `training/train.py`
- `training/callbacks.py`
- `evaluation/benchmark.py`
- `evaluation/evaluate.py`
- `evaluation/record.py`
- `evaluation/visualize.py`
- `tests/test_env.py`
- `tests/test_physics.py`
- `tests/test_reward.py`

This matters because the active roadmap should optimize for first working system, not for maximum conceptual scope.

## What COLOSSEUM Means

COLOSSEUM is not just self-play.

In full scope, COLOSSEUM is a competitive arena system built on top of Dogfight AI where:

- many agents compete instead of just one learner versus one opponent
- strength is measured relative to the population, not just against a fixed baseline
- the best agents are archived and promoted as champions
- later versions can evolve both behavior and body design
- training becomes league-driven rather than single-run driven

In practice, COLOSSEUM should be built in versions.

## Roadmap Summary

There are two tracks:

1. Dogfight Core
2. COLOSSEUM Versions

Dogfight Core must reach completion before COLOSSEUM moves beyond Version 1.

## Dogfight Core

### Goal

Produce a portfolio-quality 2D RL dogfight system with:

- smooth continuous-control combat
- reproducible PPO training
- checkpoint and resume support
- rule-based benchmarking
- multi-seed evaluation
- rendered demos and writeup assets

### Target End State

- PPO agent beats a simple rule-based baseline reliably
- self-play produces stronger behavior than baseline-only training
- evaluation scripts report win rate, reward, and episode length
- trained policies can be visualized and recorded
- runs are resumable and comparable across seeds

## Delivery Phases

### Phase 0: Simulation Foundation

Build and validate the environment layer before any serious training.

Files:

- `envs/physics.py`
- `envs/observation.py`
- `envs/reward.py`
- `envs/dogfight_env.py`
- `tests/test_physics.py`
- `tests/test_reward.py`
- `tests/test_env.py`

Done when:

- jets move correctly
- bullets spawn, travel, expire, and collide correctly
- toroidal wrapping behaves correctly
- observations stay in expected numeric ranges
- reward terms are testable and interpretable
- Gymnasium env passes `check_env()`

### Phase 1: Baseline Training

Build one complete PPO training path against a rule-based opponent.

Files:

- `agents/rule_based.py`
- `training/train.py`
- `training/callbacks.py`
- `agents/ppo_agent.py`

Done when:

- one training run completes without crashing
- checkpoints are saved
- a trained agent shows basic pursuit and firing behavior
- rule-based benchmark exists and random-policy sanity checks fail as expected

### Phase 2: Self-Play

Upgrade from fixed-opponent training to a controlled self-play league.

Files:

- `agents/self_play.py`
- `training/train.py`
- `training/callbacks.py`

Done when:

- opponent pool can save and load cleanly
- the current learner can train against frozen historical policies
- resume restores both model state and opponent-pool state
- self-play beats pure rule-based training on evaluation

### Phase 3: Evaluation and Presentation

Turn the project into something measurable and demonstrable.

Files:

- `evaluation/evaluate.py`
- `evaluation/benchmark.py`
- `evaluation/visualize.py`
- `evaluation/record.py`

Done when:

- win rate and episode metrics are reported over fixed evaluation suites
- visualizer can watch a trained fight
- recording pipeline exports GIF or MP4
- the repo can produce plots, tables, and a top-of-README demo asset

### Phase 4: Multi-Seed and Writeup

Make the claims stronger and the output more professional.

Done when:

- at least 3 seeds have been run
- final results are reported as mean plus or minus standard deviation
- notebooks or scripts can aggregate and visualize results
- the project can be explained as a rigorous RL build rather than just a demo

## COLOSSEUM Versioning

COLOSSEUM stays in the plan at full scope, but each version has a clear entry point and exit criterion.

### COLOSSEUM Version 0: Naming Only

Purpose:

- define COLOSSEUM as the project's expansion layer
- keep terminology stable early

What it is:

- not a code system yet
- just the concept that Dogfight AI will eventually become a competitive arena

Exit condition:

- Dogfight Core Phase 1 is complete

### COLOSSEUM Version 1: Checkpoint Arena

Purpose:

- turn saved PPO checkpoints into a league

What it includes:

- archive of saved models with metadata
- round-robin or sampled head-to-head matches between checkpoints
- champion selection based on evaluation results
- promotion of top checkpoints into the self-play pool

What it is not yet:

- no morphology evolution
- no population breeding
- no separate infrastructure brain

Why this matters:

- this is the smallest real COLOSSEUM
- it already gives you league dynamics, champion tracking, and stronger self-play

Exit condition:

- archived checkpoints can be evaluated against each other automatically
- a current champion can be identified and reused as an opponent

### COLOSSEUM Version 2: Population League

Purpose:

- move from checkpoint league to explicit population management

What it includes:

- multiple concurrently maintained agents
- scheduled tournaments
- ranking and champion tables
- controlled replacement of weak agents by stronger variants

How new agents are created:

- start simple
- use seed variation, hyperparameter variation, or warm-started copies with perturbations

Why this version exists:

- it creates population pressure without introducing body evolution yet

Exit condition:

- the project can maintain a persistent population
- ranking changes over time
- the champion is stronger than a single-run baseline agent

### COLOSSEUM Version 3: Full Behavioral COLOSSEUM

Purpose:

- make the arena a true competitive training ecosystem

What it includes:

- persistent agent archive
- lineage tracking
- elite promotion and retirement
- diversity monitoring
- automated tournament scheduling
- richer champion history across generations

At this point COLOSSEUM becomes:

- more than self-play
- more than checkpoint comparison
- a real league training system with historical memory

Exit condition:

- the project can produce generations of champions, not just one best model

### COLOSSEUM Version 4: Full-Scope COLOSSEUM

Purpose:

- realize the complete original vision

What it includes:

- agent as `(genome, policy)`
- morphology or body evolution
- policy warm-starting across generations
- diversity-aware selection pressure
- population metrics such as variance, lineage, and collapse detection

This is the original large-scale concept:

- bodies evolve
- behaviors evolve
- agents fight for relative fitness
- champions are archived over generations

Dependencies:

- stable Dogfight Core
- mature COLOSSEUM Version 1 through 3 systems
- enough tooling to support expensive experiments

Exit condition:

- the project can evolve both body and behavior in a competitive arena and preserve lineage over generations

## What Is Explicitly Out of Scope For Now

These ideas can stay in the broader vision, but they are not on the active roadmap:

- AKASHIC as an autonomous diagnosis layer
- ORCHESTRATOR as a general execution engine
- NEMESIS as an asymmetric destroyer-versus-champion system
- LEVIATHAN as a 1000-agent ensemble voting system
- cross-domain validation on unrelated environments

Those are future extensions, not current milestones.

## Active Build Order

This is the recommended order for the repo right now.

1. Finish `envs/physics.py` and validate with tests.
2. Lock down `envs/observation.py` and test ranges and toroidal logic.
3. Implement `envs/reward.py` with explicit reward components.
4. Build `envs/dogfight_env.py` and pass `check_env()`.
5. Implement `agents/rule_based.py`.
6. Implement `training/train.py` for one clean PPO path.
7. Add checkpointing and resume in `training/callbacks.py`.
8. Implement `agents/self_play.py`.
9. Build `evaluation/evaluate.py`, `evaluation/benchmark.py`, and `evaluation/visualize.py`.
10. Add `evaluation/record.py`.
11. Add COLOSSEUM Version 1 on top of the working checkpoint archive.
12. Only then consider COLOSSEUM Version 2 and beyond.

## Definition of Success

Short-term success:

- Dogfight AI works end to end
- one trained agent is credible
- self-play improves quality
- evaluation is reproducible

Medium-term success:

- COLOSSEUM Version 1 and Version 2 produce a meaningful league and champion system

Long-term success:

- COLOSSEUM Version 4 reaches the original full-scope vision of population-based competitive evolution over both policy and body design

## Practical Rule

If a new feature does not help us get to a stable dogfight agent or to COLOSSEUM Version 1, it is probably too early.

Build the fighter first.
Then build the arena.
