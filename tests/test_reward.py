import math

from envs.config import DogfightConfig
from envs.reward import compute_reward, potential


class MockJet:
    def __init__(self, x, y, theta=0.0, v=3.0, v_max=6.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        self.v_max = v_max


CONFIG = DogfightConfig(
    arena_width=800,
    arena_height=800,
    kill_reward=100.0,
    death_penalty=-100.0,
    hit_reward=1.0,
    hit_taken_penalty=-0.5,
    fire_cone_reward=0.3,
    closing_shaping_scale=0.5,
    shaping_gamma=1.0,
    time_penalty=-0.1,
    fire_cone_angle_deg=15.0,
)


def test_potential_decreases_with_distance() -> None:
    ego = MockJet(100, 100)

    phi_near = potential(ego, MockJet(150, 100), CONFIG)
    phi_far = potential(ego, MockJet(400, 100), CONFIG)

    assert phi_near > phi_far
    assert math.isclose(potential(ego, MockJet(100, 100), CONFIG), 0.0)


def test_potential_uses_toroidal_shortest_path() -> None:
    ego = MockJet(790, 400)
    opponent = MockJet(10, 400)

    expected = -CONFIG.closing_shaping_scale * 20.0 / CONFIG.arena_diag
    assert math.isclose(potential(ego, opponent, CONFIG), expected)


def test_compute_reward_terminal_win() -> None:
    ego = MockJet(100, 100, 0.0)
    opponent = MockJet(200, 100, 0.0)

    reward = compute_reward({"won": True}, ego, opponent, CONFIG)

    assert math.isclose(reward, 100.2)


def test_compute_reward_terminal_loss() -> None:
    ego = MockJet(100, 100, 0.0)
    opponent = MockJet(200, 100, 0.0)

    reward = compute_reward({"lost": True}, ego, opponent, CONFIG)

    assert math.isclose(reward, -99.8)


def test_compute_reward_hit_opponent() -> None:
    ego = MockJet(100, 100, 0.0)
    opponent = MockJet(200, 100, 0.0)

    reward = compute_reward({"hit_opponent": True}, ego, opponent, CONFIG)

    assert math.isclose(reward, 1.2)


def test_compute_reward_got_hit() -> None:
    ego = MockJet(100, 100, 0.0)
    opponent = MockJet(200, 100, 0.0)

    reward = compute_reward({"got_hit": True}, ego, opponent, CONFIG)

    assert math.isclose(reward, -0.3)


def test_compute_reward_fire_cone_bonus_only_when_aligned() -> None:
    ego = MockJet(100, 100, 0.0)

    aligned = compute_reward({}, ego, MockJet(200, 100, 0.0), CONFIG)
    misaligned = compute_reward({}, ego, MockJet(100, 200, 0.0), CONFIG)

    assert math.isclose(aligned, 0.2)
    assert math.isclose(misaligned, -0.1)


def test_shaping_positive_when_closing_negative_when_fleeing() -> None:
    ego = MockJet(100, 100, 0.0)
    opponent = MockJet(300, 100, 0.0)

    phi_far = potential(MockJet(100, 100), MockJet(420, 100), CONFIG)
    phi_near = potential(MockJet(100, 100), MockJet(220, 100), CONFIG)

    closed_in = compute_reward({"prev_phi": phi_far}, ego, opponent, CONFIG)
    fled = compute_reward({"prev_phi": phi_near}, ego, opponent, CONFIG)
    baseline = compute_reward({}, ego, opponent, CONFIG)

    assert closed_in > baseline > fled


def test_shaping_telescopes_to_zero_over_closed_loop() -> None:
    # With gamma=1, potential-based shaping over any loop that returns to the
    # starting state must sum to exactly 0 — no reward farming from movement.
    ego = MockJet(100, 100, 0.0)
    opponent_positions = [(300, 100), (400, 200), (250, 350), (120, 180), (300, 100)]

    total_shaping = 0.0
    for prev_pos, next_pos in zip(opponent_positions, opponent_positions[1:]):
        prev_phi = potential(ego, MockJet(*prev_pos), CONFIG)
        next_phi = potential(ego, MockJet(*next_pos), CONFIG)
        total_shaping += CONFIG.shaping_gamma * next_phi - prev_phi

    assert math.isclose(total_shaping, 0.0, abs_tol=1e-12)


def test_shaping_uses_zero_potential_at_terminal_states() -> None:
    ego = MockJet(100, 100, 0.0)
    opponent = MockJet(200, 100, 0.0)
    prev_phi = potential(ego, opponent, CONFIG)

    reward = compute_reward({"won": True, "prev_phi": prev_phi}, ego, opponent, CONFIG)

    # kill + fire cone + time penalty + (gamma * 0 - prev_phi)
    assert math.isclose(reward, 100.2 - prev_phi)
