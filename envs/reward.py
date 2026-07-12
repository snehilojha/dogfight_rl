import math


def wrapped_delta(delta, arena_size):
    if abs(delta) > arena_size / 2:
        delta = delta - math.copysign(arena_size, delta)
    return delta


def toroidal_relative_position(ego_jet, opponent_jet, config):
    dx = opponent_jet.x - ego_jet.x
    dy = opponent_jet.y - ego_jet.y

    dx = wrapped_delta(dx, config.arena_width)
    dy = wrapped_delta(dy, config.arena_height)
    return dx, dy


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

    dx, dy = toroidal_relative_position(ego_jet, opponent_jet, config)
    distance = math.sqrt(dx * dx + dy * dy)

    angle_to_opponent = math.atan2(dy, dx)
    rel_angle = angle_to_opponent - ego_jet.theta
    rel_angle = (rel_angle + math.pi) % (2 * math.pi) - math.pi

    if abs(rel_angle) <= math.radians(config.fire_cone_angle_deg):
        reward += config.fire_cone_reward

    prev_distance = events.get("prev_distance")
    if prev_distance is not None and prev_distance > config.close_range_distance and distance < prev_distance:
        reward += config.closing_distance_reward

    if config.max_speed > 0:
        reward += config.speed_reward_scale * (ego_jet.v / config.max_speed)

    reward += config.time_penalty

    return reward
