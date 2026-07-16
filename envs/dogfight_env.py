import math

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from envs.config import DogfightConfig
from envs.observation import build_obs
from envs.physics import Bullet, Jet, check_collisions, toroidal_delta
from envs.reward import compute_reward, potential


class DogfightEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config=None, opponent_policy=None, opponent_provider=None, render_mode=None):
        super().__init__()

        if config is None:
            self.config = DogfightConfig()
        elif isinstance(config, DogfightConfig):
            self.config = config
        else:
            self.config = DogfightConfig.from_dict(config)

        self.render_mode = render_mode
        # opponent_policy is the fixed opponent; opponent_provider (used by
        # self-play) is a zero-arg callable resampled each reset that returns
        # the policy for the coming episode and takes precedence when set.
        self.opponent_policy = opponent_policy
        self.opponent_provider = opponent_provider

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
        self.font = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.bullets = []
        self.next_bullet_id = 0

        if self.opponent_provider is not None:
            self.opponent_policy = self.opponent_provider()

        (ego_x, ego_y, ego_theta), (opp_x, opp_y, opp_theta) = self._sample_spawns()
        self.ego_jet = self._make_jet(ego_x, ego_y, ego_theta, jet_id=0)
        self.opponent_jet = self._make_jet(opp_x, opp_y, opp_theta, jet_id=1)

        obs = build_obs(self.ego_jet, self.opponent_jet, self.config)
        info = {}
        return obs, info

    def _make_jet(self, x, y, theta, jet_id):
        return Jet(
            x=x,
            y=y,
            theta=theta,
            id=jet_id,
            v_min=self.config.min_speed,
            v_max=self.config.max_speed,
            w_max=self.config.max_turn_rate,
            arena_width=self.config.arena_width,
            arena_height=self.config.arena_height,
            max_health=self.config.max_health,
            radius=self.config.jet_radius,
        )

    def _sample_spawns(self):
        """Return ((x, y, theta), (x, y, theta)) spawn poses for ego and opponent.

        Randomized spawns draw uniform positions (rejection-sampled to keep a
        minimum toroidal separation) and uniform headings from the env's seeded
        RNG, so a fixed seed reproduces the layout. Deterministic spawns place
        both jets at facing quarter-points.
        """
        arena_w = self.config.arena_width
        arena_h = self.config.arena_height

        if not self.config.randomize_spawns:
            return (arena_w * 0.25, arena_h * 0.5, 0.0), (arena_w * 0.75, arena_h * 0.5, math.pi)

        rng = self.np_random
        min_sep = self.config.min_spawn_separation
        ego_x = float(rng.uniform(0.0, arena_w))
        ego_y = float(rng.uniform(0.0, arena_h))

        for _ in range(1000):
            opp_x = float(rng.uniform(0.0, arena_w))
            opp_y = float(rng.uniform(0.0, arena_h))
            dx, dy = toroidal_delta(ego_x, ego_y, opp_x, opp_y, arena_w, arena_h)
            if math.hypot(dx, dy) >= min_sep:
                break

        ego_theta = float(rng.uniform(-math.pi, math.pi))
        opp_theta = float(rng.uniform(-math.pi, math.pi))
        return (ego_x, ego_y, ego_theta), (opp_x, opp_y, opp_theta)

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
        if not pygame.font.get_init():
            pygame.font.init()
        self.font = pygame.font.SysFont("consolas", 16)
        size = (int(self.config.arena_width), int(self.config.arena_height))
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode(size)
        else:
            self.screen = pygame.Surface(size)
        self.clock = pygame.time.Clock()

    def _draw_scene(self):
        self.screen.fill((18, 22, 32))

        ego_color = (80, 220, 120)
        opp_color = (220, 90, 90)
        if self.ego_jet is not None:
            self._draw_jet(self.ego_jet, ego_color)
            self._draw_health_bar(self.ego_jet, ego_color)
        if self.opponent_jet is not None:
            self._draw_jet(self.opponent_jet, opp_color)
            self._draw_health_bar(self.opponent_jet, opp_color)

        for bullet in self.bullets:
            pygame.draw.circle(self.screen, (255, 230, 120), (int(bullet.x), int(bullet.y)), int(bullet.radius))

        if self.font is not None:
            label = self.font.render(f"step {self.step_count}", True, (200, 210, 225))
            self.screen.blit(label, (8, 8))

    def _draw_health_bar(self, jet, color):
        if self.font is None:
            return
        frac = max(0.0, jet.health / self.config.max_health)
        width, height = 30, 4
        x = int(jet.x - width / 2)
        y = int(jet.y - jet.radius - 10)
        pygame.draw.rect(self.screen, (60, 60, 70), (x, y, width, height))
        pygame.draw.rect(self.screen, color, (x, y, int(width * frac), height))

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
