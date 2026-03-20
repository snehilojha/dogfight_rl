import numpy as np
import math

def build_obs(ego_jet, opponent_jet, config) -> np.ndarray:
    """
    Build observation vector for PPO agent.
    
    Args:
        ego_jet: Ego jet object with attributes: x, y, heading, speed, gun_cooldown, health
        opponent_jet: Opponent jet object with attributes: x, y, heading
    
    Returns:
        np.ndarray: 14-dimensional observation vector
    """
    arena_w = config.get('arena_width')
    arena_h = config.get('arena_height')
    arena_diag = config.get('arena_diag')
    max_speed = config.get('max_speed')
    max_cooldown = config.get('max_cooldown')
    max_health = config.get('max_health')

    obs = np.zeros(14, dtype=np.float32)
    obs[0]  = ego_jet.x / arena_w * 2 - 1
    obs[1]  = ego_jet.y / arena_h * 2 - 1
    obs[2]  = math.sin(ego_jet.theta)
    obs[3]  = math.cos(ego_jet.theta)
    obs[4]  = ego_jet.v / max_speed

    # Toroidal (wrap-around) boundaries
    dx = opponent_jet.x - ego_jet.x
    dy = opponent_jet.y - ego_jet.y
    
    if abs(dx) > arena_w / 2:
        dx = dx - math.copysign(arena_w, dx)
    
    if abs(dy) > arena_h / 2:
        dy = dy - math.copysign(arena_h, dy)

    # === OPPONENT RELATIVE STATE ===
    obs[5]  = dx / arena_w
    obs[6]  = dy / arena_h

    distance = math.sqrt(dx*dx + dy*dy)
    obs[7]  = distance / arena_diag

    angle_to_opponent = math.atan2(dy, dx)
    rel_angle = angle_to_opponent - ego_jet.theta
    rel_angle = (rel_angle + math.pi) % (2 * math.pi) - math.pi

    obs[8]  = math.sin(rel_angle)
    obs[9]  = math.cos(rel_angle)

    obs[10] = math.sin(opponent_jet.theta)
    obs[11] = math.cos(opponent_jet.theta)

    # === TACTICAL STATE ===
    obs[12] = ego_jet.gun_cooldown / max_cooldown
    obs[13] = ego_jet.health / max_health
    return obs
    