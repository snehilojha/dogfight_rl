import math


def wrapped_delta(delta, arena_size):
    if abs(delta) > arena_size / 2:
        delta = delta - math.copysign(arena_size, delta)
    return delta


def toroidal_relative_position(ego_jet, opponent_jet, config):
    arena_w = config["arena_width"]
    arena_h = config["arena_height"]

    dx = opponent_jet.x - ego_jet.x
    dy = opponent_jet.y - ego_jet.y

    dx = wrapped_delta(dx, arena_w)
    dy = wrapped_delta(dy, arena_h)
    return dx, dy


def compute_reward(events, ego_jet, opponent_jet, config):
    reward = 0.0

    kill_reward = config.get("kill_reward", 100.0)
    death_penalty = config.get("death_penalty", -100.0)
    hit_reward = config.get("hit_reward", 1.0)
    hit_taken_penalty = config.get("hit_taken_penalty", -0.5)
    fire_cone_reward = config.get("fire_cone_reward", 0.3)
    closing_distance_reward = config.get("closing_distance_reward", 0.2)
    speed_reward_scale = config.get("speed_reward_scale", 0.1)
    time_penalty = config.get("time_penalty", -0.1)
    out_of_bounds_penalty = config.get("out_of_bounds_penalty", -0.2)
    fire_cone_angle_deg = config.get("fire_cone_angle_deg", 15.0)
    close_range_dist = config.get("close_range_distance", 150.0)

    if events.get("won", False):
        reward += kill_reward

    if events.get("lost", False):
        reward += death_penalty

    if events.get("hit_opponent", False):
        reward += hit_reward

    if events.get("got_hit", False):
        reward += hit_taken_penalty

    dx, dy = toroidal_relative_position(ego_jet, opponent_jet, config)
    distance = math.sqrt(dx * dx + dy * dy)

    angle_to_opponent = math.atan2(dy, dx)
    rel_angle = angle_to_opponent - ego_jet.theta
    rel_angle = (rel_angle + math.pi) % (2 * math.pi) - math.pi

    fire_cone_angle_rad = math.radians(fire_cone_angle_deg)
    if abs(rel_angle) <= fire_cone_angle_rad:
        reward += fire_cone_reward

    prev_distance = events.get("prev_distance")
    if prev_distance is not None and prev_distance > close_range_dist and distance < prev_distance:
        reward += closing_distance_reward

    max_speed = config.get("max_speed", ego_jet.v_max)
    if max_speed > 0:
        reward += speed_reward_scale * (ego_jet.v / max_speed)

    reward += time_penalty

    if events.get("out_of_bounds", False):
        reward += out_of_bounds_penalty

    return reward
