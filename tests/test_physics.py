import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.physics import Bullet, Jet, check_collisions, wrap_angle, wrapped_delta


ARENA_W = 800
ARENA_H = 800


def make_jet(**overrides) -> Jet:
    params = {
        "x": 100.0,
        "y": 100.0,
        "theta": 0.0,
        "id": 0,
        "v_min": 1.0,
        "v_max": 6.0,
        "w_max": math.radians(4),
        "arena_width": ARENA_W,
        "arena_height": ARENA_H,
    }
    params.update(overrides)
    return Jet(**params)


def make_bullet(**overrides) -> Bullet:
    params = {
        "x": 100.0,
        "y": 100.0,
        "theta": 0.0,
        "owner_id": 0,
        "id": 0,
        "arena_width": ARENA_W,
        "arena_height": ARENA_H,
        "bullet_speed": 12.0,
    }
    params.update(overrides)
    return Bullet(**params)


def test_wrap_angle_normalizes_to_expected_range() -> None:
    wrapped = wrap_angle(3 * math.pi)
    assert -math.pi <= wrapped < math.pi
    assert math.isclose(wrapped, -math.pi)


def test_wrapped_delta_prefers_toroidal_short_path() -> None:
    assert wrapped_delta(790.0, 800.0) == -10.0
    assert wrapped_delta(-790.0, 800.0) == 10.0


def test_jet_update_clamps_action_and_moves_forward() -> None:
    jet = make_jet()
    jet.update((2.0, 3.0, -1.0))

    assert jet.last_action == (1.0, 1.0, 0.0)
    assert math.isclose(jet.theta, math.radians(4), rel_tol=1e-6)
    assert math.isclose(jet.v, 6.0)
    assert jet.x > 100.0


def test_jet_wraps_around_arena_edges() -> None:
    jet = make_jet(x=799.5, y=799.5, theta=0.0)
    jet.update((0.0, 1.0, 0.0))

    assert 0.0 <= jet.x < 10.0
    assert 0.0 <= jet.y < ARENA_H


def test_jet_gun_cooldown_never_goes_below_zero() -> None:
    jet = make_jet()
    jet.gun_cooldown = 1

    jet.update((0.0, 0.0, 0.0))
    assert jet.gun_cooldown == 0

    jet.update((0.0, 0.0, 0.0))
    assert jet.gun_cooldown == 0


def test_bullet_moves_and_expires_after_lifetime() -> None:
    bullet = make_bullet()
    bullet.lifetime = 2

    bullet.update()
    assert bullet.alive is True
    assert bullet.lifetime == 1

    bullet.update()
    assert bullet.alive is False
    assert bullet.lifetime == 0


def test_bullet_wraps_around_arena_edges() -> None:
    bullet = make_bullet(x=799.0, y=400.0, theta=0.0)
    bullet.update()

    assert 0.0 <= bullet.x < 20.0


def test_check_collisions_detects_hit() -> None:
    shooter = make_jet(id=0, x=400.0, y=400.0)
    target = make_jet(id=1, x=420.0, y=400.0)
    bullet = make_bullet(x=417.0, y=400.0, owner_id=0, id=7)

    result = check_collisions([shooter, target], [bullet])

    assert result["bullet_hits"] == [(7, 1, 0)]


def test_check_collisions_ignores_self_hits() -> None:
    shooter = make_jet(id=0, x=400.0, y=400.0)
    bullet = make_bullet(x=400.0, y=400.0, owner_id=0, id=3)

    result = check_collisions([shooter], [bullet])

    assert result["bullet_hits"] == []


def test_check_collisions_respects_toroidal_distance() -> None:
    shooter = make_jet(id=0, x=400.0, y=400.0)
    target = make_jet(id=1, x=2.0, y=400.0)
    bullet = make_bullet(x=798.0, y=400.0, owner_id=0, id=5)

    result = check_collisions([shooter, target], [bullet])

    assert result["bullet_hits"] == [(5, 1, 0)]


def test_dead_bullets_and_dead_jets_do_not_collide() -> None:
    shooter = make_jet(id=0)
    dead_target = make_jet(id=1)
    dead_target.alive = False
    dead_bullet = make_bullet(owner_id=0, id=9)
    dead_bullet.alive = False

    result = check_collisions([shooter, dead_target], [dead_bullet])

    assert result["bullet_hits"] == []
