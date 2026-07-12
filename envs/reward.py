import math

from envs.physics import relative_bearing, toroidal_delta


def potential(ego_jet, opponent_jet, config):
    """Distance-based potential phi(s) for potential-based reward shaping.

    More negative when far from the opponent, 0 at zero distance. Shaping is
    gamma * phi(s') - phi(s), which is policy-invariant (Ng et al. 1999) as
    long as gamma matches the discount used by the learner.
    """
    dx, dy = toroidal_delta(
        ego_jet.x, ego_jet.y, opponent_jet.x, opponent_jet.y, config.arena_width, config.arena_height
    )
    distance = math.hypot(dx, dy)
    return -config.closing_shaping_scale * distance / config.arena_diag


def compute_reward(events, ego_jet, opponent_jet, config):
    reward = 0.0

    if events.get("won", False):
        reward += config.kill_reward

    if events.get("lost", False):
        reward += config.death_penalty

    if events.get("hit_opponent", False):
        reward += config.hit_reward

    if events.get("got_hit", False):
        reward += config.hit_taken_penalty

    dx, dy = toroidal_delta(
        ego_jet.x, ego_jet.y, opponent_jet.x, opponent_jet.y, config.arena_width, config.arena_height
    )
    rel_angle = relative_bearing(dx, dy, ego_jet.theta)

    if abs(rel_angle) <= math.radians(config.fire_cone_angle_deg):
        reward += config.fire_cone_reward

    prev_phi = events.get("prev_phi")
    if prev_phi is not None:
        # phi(terminal) must be 0 for the shaping to stay policy-invariant.
        if events.get("won", False) or events.get("lost", False):
            next_phi = 0.0
        else:
            next_phi = potential(ego_jet, opponent_jet, config)
        reward += config.shaping_gamma * next_phi - prev_phi

    reward += config.time_penalty

    return reward
