import numpy as np
import math
from envs.observation import build_obs

# Test the observation function
config = {
    'arena_width': 800,
    'arena_height': 800,
    'max_speed': 6,
    'max_cooldown': 20,
    'max_health': 100,
    'arena_diag': 1131.370849898476
}

# Create mock jet objects
class MockJet:
    def __init__(self, x, y, theta, v, gun_cooldown, health):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        self.gun_cooldown = gun_cooldown
        self.health = health

ego_jet = MockJet(x=400, y=400, theta=0, v=3, gun_cooldown=0, health=100)
opponent_jet = MockJet(x=500, y=500, theta=0, v=3, gun_cooldown=0, health=100)

obs = build_obs(ego_jet, opponent_jet, config)
print("Observation shape:", obs.shape)
print("Observation:", obs)

#replace manual print statements with assert
assert obs[0] == 400 / 800 * 2 - 1
assert obs[1] == 400 / 800 * 2 - 1
assert obs[2] == math.sin(0)
assert obs[3] == math.cos(0)
assert obs[4] == 3 / 6
assert obs[5] == (500 - 400) / 800
assert obs[6] == (500 - 400) / 800
assert obs[7] == math.sqrt((500 - 400)**2 + (500 - 400)**2) / 1131.370849898476
assert obs[8] == math.sin(math.atan2(100, 100) - 0)
assert obs[9] == math.cos(math.atan2(100, 100) - 0)
assert obs[10] == math.sin(0)
assert obs[11] == math.cos(0)
assert obs[12] == 0 / 20
assert obs[13] == 100 / 100
print("All assertions passed!")