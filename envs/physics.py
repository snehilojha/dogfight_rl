import math

class Jet:
    def __init__(self, x, y, theta, id,v_min, v_max, w_max, arena_width, arena_height):
        self.x = x
        self.y = y
        self.v = v_min
        self.theta = theta
        self.health = 100
        self.gun_cooldown = 0
        self.radius = 10
        self.alive = True
        self.id = id
        self.last_action = None
        self.v_min = v_min
        self.v_max = v_max
        self.w_max = w_max
        self.arena_width = arena_width
        self.arena_height = arena_height

    def update(self, action):
        self.last_action = action
        a0, a1, a2 = action
        a0 = max(-1.0, min(1.0, a0))
        a1 = max(0.0, min(1.0, a1))
        a2 = max(0.0, min(1.0, a2))
        w = a0 * self.w_max
        self.theta += w
        self.theta = (self.theta + math.pi) % (2 * math.pi) - math.pi

        self.v = self.v_min + a1 * (self.v_max - self.v_min)
        vx = self.v * math.cos(self.theta)
        vy = self.v * math.sin(self.theta)
        
        self.x += vx
        self.y += vy

        self.x = self.x % self.arena_width
        self.y = self.y % self.arena_height

        self.gun_cooldown = max(0, self.gun_cooldown - 1)


class Bullet:
    def __init__(self, x, y, theta, owner_id,id, arena_width, arena_height, bullet_speed):
        self.x = x
        self.y = y
        self.theta = theta
        self.owner_id = owner_id
        self.bullet_speed = bullet_speed
        self.lifetime = 60
        self.id = id
        self.radius = 3
        self.alive = True
        self.arena_width = arena_width
        self.arena_height = arena_height

    def update(self):
        self.x += self.bullet_speed * math.cos(self.theta)
        self.y += self.bullet_speed * math.sin(self.theta)
        self.lifetime -= 1
        self.alive = self.lifetime > 0
        self.x = self.x % self.arena_width
        self.y = self.y % self.arena_height

def check_collisions(jets, bullets):
    bullet_hits = []

    for bullet in bullets:
        if not bullet.alive:
            continue

        for jet in jets:
            if not jet.alive:
                continue

            # prevent self-hit
            if jet.id == bullet.owner_id:
                continue

            # --- Toroidal distance ---
            dx = abs(bullet.x - jet.x)
            dy = abs(bullet.y - jet.y)

            dx = min(dx, bullet.arena_width - dx)
            dy = min(dy, bullet.arena_height - dy)

            # --- Collision check ---
            r_sum = bullet.radius + jet.radius

            if (dx * dx + dy * dy) <= (r_sum * r_sum):
                bullet_hits.append(
                    (bullet.id, jet.id, bullet.owner_id)
                )

                break  # one bullet hits only one jet

    return {
        "bullet_hits": bullet_hits
    }