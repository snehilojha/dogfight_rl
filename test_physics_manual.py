"""
test_physics_manual.py — throwaway sanity check for physics.py.

Scenario:
    - Jet 0 at (400, 400), heading east (theta=0)
    - Jet 1 at (500, 400), heading west (theta=pi)  <- target
    - Bullet fired by Jet 0 aimed directly east (theta=0), spawned just in
      front of Jet 0 at (415, 400) so it travels toward Jet 1.

Expected: after a few update steps, check_collisions reports a hit on Jet 1.
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from envs.physics import Jet, Bullet, check_collisions

ARENA_W = 800
ARENA_H = 800
BULLET_SPEED = 12

jet0 = Jet(
    x=400, y=400, theta=0.0, id=0,
    v_min=1, v_max=6, w_max=math.radians(4),
    arena_width=ARENA_W, arena_height=ARENA_H,
)

jet1 = Jet(
    x=500, y=400, theta=math.pi, id=1,
    v_min=1, v_max=6, w_max=math.radians(4),
    arena_width=ARENA_W, arena_height=ARENA_H,
)

# Bullet owned by jet0, aimed east, spawned just ahead of jet0
bullet = Bullet(
    x=415, y=400, theta=0.0,
    owner_id=0, id=0,
    arena_width=ARENA_W, arena_height=ARENA_H,
    bullet_speed=BULLET_SPEED,
)

jets = [jet0, jet1]
bullets = [bullet]

print("=== Initial state ===")
print(f"  Jet0   pos=({jet0.x:.1f}, {jet0.y:.1f})")
print(f"  Jet1   pos=({jet1.x:.1f}, {jet1.y:.1f})")
print(f"  Bullet pos=({bullet.x:.1f}, {bullet.y:.1f})  alive={bullet.alive}")

hit_registered = False

for step in range(1, 11):
    bullet.update()

    result = check_collisions(jets, bullets)

    print(f"\nStep {step}: bullet pos=({bullet.x:.1f}, {bullet.y:.1f})  alive={bullet.alive}")

    if result["bullet_hits"]:
        bid, jid, oid = result["bullet_hits"][0]
        print(f"  HIT  bullet_id={bid} hit jet_id={jid} (fired by owner_id={oid})")
        # step() would do this mutation — we replicate it here to stop re-hitting
        bullet.alive = False
        hit_registered = True
        break
    else:
        print("  No hit yet.")

print()
if hit_registered:
    print("PASS — collision detected as expected.")
else:
    print("FAIL — no collision detected after 10 steps.")
