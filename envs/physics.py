import math


def wrap_angle(theta):
    return (theta + math.pi) % (2 * math.pi) - math.pi


def wrapped_delta(delta, arena_size):
    if abs(delta) > arena_size / 2:
        delta = delta - math.copysign(arena_size, delta)
    return delta


class Jet:
    def __init__(self, x, y, theta, id, v_min, v_max, w_max, arena_width, arena_height):
        self.x = x
        self.y = y
        self.theta = wrap_angle(theta)
        self.id = id

        self.v_min = v_min
        self.v_max = v_max
        self.w_max = w_max
        self.v = v_min

        self.arena_width = arena_width
        self.arena_height = arena_height

        self.health = 100
        self.gun_cooldown = 0
        self.radius = 10
        self.alive = True
        self.last_action = None

    def update(self, action):
        turn, throttle, fire = action

        turn = max(-1.0, min(1.0, float(turn)))
        throttle = max(0.0, min(1.0, float(throttle)))
        fire = max(0.0, min(1.0, float(fire)))
        self.last_action = (turn, throttle, fire)

        self.theta = wrap_angle(self.theta + turn * self.w_max)
        self.v = self.v_min + throttle * (self.v_max - self.v_min)

        self.x += self.v * math.cos(self.theta)
        self.y += self.v * math.sin(self.theta)

        self.x = self.x % self.arena_width
        self.y = self.y % self.arena_height

        self.gun_cooldown = max(0, self.gun_cooldown - 1)


class Bullet:
    def __init__(self, x, y, theta, owner_id, id, arena_width, arena_height, bullet_speed):
        self.x = x
        self.y = y
        self.theta = wrap_angle(theta)
        self.owner_id = owner_id
        self.id = id

        self.arena_width = arena_width
        self.arena_height = arena_height
        self.bullet_speed = bullet_speed

        self.lifetime = 60
        self.radius = 3
        self.alive = True

    def update(self):
        if not self.alive:
            return

        self.x += self.bullet_speed * math.cos(self.theta)
        self.y += self.bullet_speed * math.sin(self.theta)

        self.x = self.x % self.arena_width
        self.y = self.y % self.arena_height

        self.lifetime -= 1
        self.alive = self.lifetime > 0


def check_collisions(jets, bullets):
    bullet_hits = []

    for bullet in bullets:
        if not bullet.alive:
            continue

        for jet in jets:
            if not jet.alive:
                continue

            if jet.id == bullet.owner_id:
                continue

            dx = wrapped_delta(bullet.x - jet.x, bullet.arena_width)
            dy = wrapped_delta(bullet.y - jet.y, bullet.arena_height)
            r_sum = bullet.radius + jet.radius

            if dx * dx + dy * dy <= r_sum * r_sum:
                bullet_hits.append((bullet.id, jet.id, bullet.owner_id))
                break

    return {"bullet_hits": bullet_hits}
