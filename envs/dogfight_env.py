import math

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from envs.observation import build_obs
from envs.physics import Bullet, Jet
from envs.reward import compute_reward, toroidal_relative_position


class DogfightEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config=None, opponent_policy=None, render_mode=None):
        super().__init__()

        base_config = {
            "arena_width": 800,
            "arena_height": 800,
            "arena_diag": math.sqrt(800**2 + 800**2),
            "max_speed": 6.0,
            "min_speed": 1.0,
            "max_turn_rate": math.radians(4),
            "bullet_speed": 12.0,
            "bullet_lifetime": 60,
            "max_cooldown": 20,
            "max_health": 100,
            "hit_damage": 25,
            "max_steps": 2000,
        }
        if config:
            base_config.update(config)
        self.config = base_config

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

        arena_w = self.config["arena_width"]
        arena_h = self.config["arena_height"]

        self.ego_jet = Jet(
            x=arena_w * 0.25,
            y=arena_h * 0.5,
            theta=0.0,
            id=0,
            v_min=self.config["min_speed"],
            v_max=self.config["max_speed"],
            w_max=self.config["max_turn_rate"],
            arena_width=arena_w,
            arena_height=arena_h,
        )
        self.opponent_jet = Jet(
            x=arena_w * 0.75,
            y=arena_h * 0.5,
            theta=math.pi,
            id=1,
            v_min=self.config["min_speed"],
            v_max=self.config["max_speed"],
            w_max=self.config["max_turn_rate"],
            arena_width=arena_w,
            arena_height=arena_h,
        )

        obs = build_obs(self.ego_jet, self.opponent_jet, self.config)
        info = {}
        return obs, info

    def step(self, action):
        self.step_count += 1

        action = np.asarray(action, dtype=np.float32)
        prev_dx, prev_dy = toroidal_relative_position(self.ego_jet, self.opponent_jet, self.config)
        prev_distance = math.sqrt(prev_dx * prev_dx + prev_dy * prev_dy)

        opponent_action = self._get_opponent_action()

        self.ego_jet.update(action)
        self.opponent_jet.update(opponent_action)

        self._maybe_fire(self.ego_jet, action)
        self._maybe_fire(self.opponent_jet, opponent_action)
        self._update_bullets()

        events = {"prev_distance": prev_distance}
        self._apply_bullet_hits(events)

        terminated = False
        truncated = False

        if not self.opponent_jet.alive:
            events["won"] = True
            terminated = True

        if not self.ego_jet.alive:
            events["lost"] = True
            terminated = True

        if self.step_count >= self.config["max_steps"]:
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
            arena_width=self.config["arena_width"],
            arena_height=self.config["arena_height"],
            bullet_speed=self.config["bullet_speed"],
        )
        bullet.lifetime = self.config["bullet_lifetime"]
        self.bullets.append(bullet)
        self.next_bullet_id += 1
        jet.gun_cooldown = self.config["max_cooldown"]

    def _update_bullets(self):
        for bullet in self.bullets:
            bullet.update()
        self.bullets = [bullet for bullet in self.bullets if bullet.alive]

    def _apply_bullet_hits(self, events):
        remaining_bullets = []

        for bullet in self.bullets:
            hit_target = None

            if self.ego_jet.alive and bullet.owner_id != self.ego_jet.id:
                if self._bullet_hits_jet(bullet, self.ego_jet):
                    hit_target = self.ego_jet
                    events["got_hit"] = True

            if hit_target is None and self.opponent_jet.alive and bullet.owner_id != self.opponent_jet.id:
                if self._bullet_hits_jet(bullet, self.opponent_jet):
                    hit_target = self.opponent_jet
                    events["hit_opponent"] = True

            if hit_target is not None:
                hit_target.health -= self.config["hit_damage"]
                if hit_target.health <= 0:
                    hit_target.health = 0
                    hit_target.alive = False
                bullet.alive = False
            else:
                remaining_bullets.append(bullet)

        self.bullets = remaining_bullets

    def _bullet_hits_jet(self, bullet, jet):
        dx = bullet.x - jet.x
        dy = bullet.y - jet.y

        if abs(dx) > bullet.arena_width / 2:
            dx = dx - math.copysign(bullet.arena_width, dx)
        if abs(dy) > bullet.arena_height / 2:
            dy = dy - math.copysign(bullet.arena_height, dy)

        r_sum = bullet.radius + jet.radius
        return dx * dx + dy * dy <= r_sum * r_sum

    def _ensure_pygame(self):
        if self.render_mode not in ("human", "rgb_array"):
            return

        if self.screen is not None:
            return

        pygame.init()
        size = (int(self.config["arena_width"]), int(self.config["arena_height"]))
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
