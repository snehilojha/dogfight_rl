import math

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from envs.config import DogfightConfig
from envs.observation import build_obs
from envs.physics import Bullet, Jet, check_collisions
from envs.reward import compute_reward, potential


class DogfightEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config=None, opponent_policy=None, render_mode=None):
        super().__init__()

        if config is None:
            self.config = DogfightConfig()
        elif isinstance(config, DogfightConfig):
            self.config = config
        else:
            self.config = DogfightConfig.from_dict(config)

        self.render_mode = render_mode
        self.opponent_policy = opponent_policy

        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(14,),
            dtype=np.float32,
        )

        self.ego_jet = None
        self.opponent_jet = None
        self.bullets = []
        self.next_bullet_id = 0
        self.step_count = 0
        self.screen = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.bullets = []
        self.next_bullet_id = 0

        arena_w = self.config.arena_width
        arena_h = self.config.arena_height

        self.ego_jet = Jet(
            x=arena_w * 0.25,
            y=arena_h * 0.5,
            theta=0.0,
            id=0,
            v_min=self.config.min_speed,
            v_max=self.config.max_speed,
            w_max=self.config.max_turn_rate,
            arena_width=arena_w,
            arena_height=arena_h,
            max_health=self.config.max_health,
            radius=self.config.jet_radius,
        )
        self.opponent_jet = Jet(
            x=arena_w * 0.75,
            y=arena_h * 0.5,
            theta=math.pi,
            id=1,
            v_min=self.config.min_speed,
            v_max=self.config.max_speed,
            w_max=self.config.max_turn_rate,
            arena_width=arena_w,
            arena_height=arena_h,
            max_health=self.config.max_health,
            radius=self.config.jet_radius,
        )

        obs = build_obs(self.ego_jet, self.opponent_jet, self.config)
        info = {}
        return obs, info

    def step(self, action):
        self.step_count += 1

        action = np.asarray(action, dtype=np.float32)
        prev_phi = potential(self.ego_jet, self.opponent_jet, self.config)

        opponent_action = self._get_opponent_action()

        self.ego_jet.update(action)
        self.opponent_jet.update(opponent_action)

        self._maybe_fire(self.ego_jet, action)
        self._maybe_fire(self.opponent_jet, opponent_action)
        self._update_bullets()

        events = {"prev_phi": prev_phi}
        self._apply_bullet_hits(events)

        terminated = False
        truncated = False

        if not self.opponent_jet.alive:
            events["won"] = True
            terminated = True

        if not self.ego_jet.alive:
            events["lost"] = True
            terminated = True

        if self.step_count >= self.config.max_steps:
            truncated = True

        reward = compute_reward(events, self.ego_jet, self.opponent_jet, self.config)
        obs = build_obs(self.ego_jet, self.opponent_jet, self.config)
        info = {
            "ego_health": self.ego_jet.health,
            "opponent_health": self.opponent_jet.health,
            "bullets_alive": len(self.bullets),
            "events": events,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        self._ensure_pygame()
        self._draw_scene()

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None

        if self.render_mode == "rgb_array":
            frame = pygame.surfarray.array3d(self.screen)
            return np.transpose(frame, (1, 0, 2))
        return None

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
        return None

    def _get_opponent_action(self):
        if self.opponent_policy is not None:
            return self.opponent_policy(self.opponent_jet, self.ego_jet, self.config)
        return np.array([0.0, 0.5, 0.0], dtype=np.float32)

    def _maybe_fire(self, jet, action):
        fire = float(action[2])
        if fire <= 0.5 or jet.gun_cooldown > 0 or not jet.alive:
            return

        bullet = Bullet(
            x=jet.x + (jet.radius + 5) * math.cos(jet.theta),
            y=jet.y + (jet.radius + 5) * math.sin(jet.theta),
            theta=jet.theta,
            owner_id=jet.id,
            id=self.next_bullet_id,
            arena_width=self.config.arena_width,
            arena_height=self.config.arena_height,
            bullet_speed=self.config.bullet_speed,
            lifetime=self.config.bullet_lifetime,
            radius=self.config.bullet_radius,
        )
        self.bullets.append(bullet)
        self.next_bullet_id += 1
        jet.gun_cooldown = self.config.max_cooldown

    def _update_bullets(self):
        for bullet in self.bullets:
            bullet.update()
        self.bullets = [bullet for bullet in self.bullets if bullet.alive]

    def _apply_bullet_hits(self, events):
        jets = {jet.id: jet for jet in (self.ego_jet, self.opponent_jet)}
        hits = check_collisions(list(jets.values()), self.bullets)["bullet_hits"]

        consumed = set()
        for bullet_id, jet_id, _owner_id in hits:
            target = jets[jet_id]
            if not target.alive:
                continue

            consumed.add(bullet_id)
            events["got_hit" if target is self.ego_jet else "hit_opponent"] = True

            target.health -= self.config.hit_damage
            if target.health <= 0:
                target.health = 0
                target.alive = False

        self.bullets = [bullet for bullet in self.bullets if bullet.id not in consumed]

    def _ensure_pygame(self):
        if self.render_mode not in ("human", "rgb_array"):
            return

        if self.screen is not None:
            return

        pygame.init()
        size = (int(self.config.arena_width), int(self.config.arena_height))
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode(size)
        else:
            self.screen = pygame.Surface(size)
        self.clock = pygame.time.Clock()

    def _draw_scene(self):
        self.screen.fill((18, 22, 32))

        if self.ego_jet is not None:
            self._draw_jet(self.ego_jet, (80, 220, 120))
        if self.opponent_jet is not None:
            self._draw_jet(self.opponent_jet, (220, 90, 90))

        for bullet in self.bullets:
            pygame.draw.circle(self.screen, (255, 230, 120), (int(bullet.x), int(bullet.y)), int(bullet.radius))

    def _draw_jet(self, jet, color):
        nose = (
            jet.x + 14 * math.cos(jet.theta),
            jet.y + 14 * math.sin(jet.theta),
        )
        left = (
            jet.x + 10 * math.cos(jet.theta + 2.5),
            jet.y + 10 * math.sin(jet.theta + 2.5),
        )
        right = (
            jet.x + 10 * math.cos(jet.theta - 2.5),
            jet.y + 10 * math.sin(jet.theta - 2.5),
        )
        pygame.draw.polygon(self.screen, color, [(int(nose[0]), int(nose[1])), (int(left[0]), int(left[1])), (int(right[0]), int(right[1]))])
