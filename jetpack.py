import random
import sys
import math
import pygame

# -----------------------------
# Config
# -----------------------------
WIDTH, HEIGHT = 960, 540
FPS = 60

BG_COLOR = (18, 18, 24)
UI_COLOR = (230, 230, 240)

GROUND_MARGIN = 24
PLAY_TOP = GROUND_MARGIN
PLAY_BOTTOM = HEIGHT - GROUND_MARGIN

# Player physics
PLAYER_W, PLAYER_H = 38, 48
PLAYER_X = 140
GRAVITY = 1600.0
THRUST = 3000.0
MAX_FALL_SPEED = 950.0
MAX_RISE_SPEED = 650.0

# World speed
SCROLL_SPEED = 320.0
SPEED_RAMP = 8.0

# Zappers
ZAPPER_MIN_LEN = 140
ZAPPER_MAX_LEN = 280
ZAPPER_THICKNESS = 14
ZAPPER_SPAWN_EVERY = (1.1, 1.8)

# Missiles
MISSILE_SPAWN_EVERY = (2.0, 3.8)
MISSILE_WARNING_TIME = 0.75
MISSILE_SPEED = 650.0
MISSILE_W, MISSILE_H = 44, 18
MISSILE_Y_JITTER = 10

PARTICLE_RATE = 80


# -----------------------------
# Helpers
# -----------------------------
def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def draw_text(surface, text, pos, font, color=UI_COLOR):
    img = font.render(text, True, color)
    surface.blit(img, pos)


def rects_collide(a: pygame.Rect, b: pygame.Rect) -> bool:
    return a.colliderect(b)


def dist_point_to_segment_sq(px, py, ax, ay, bx, by):
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    ab_len_sq = abx * abx + aby * aby

    if ab_len_sq <= 1e-9:
        dx = px - ax
        dy = py - ay
        return dx * dx + dy * dy

    t = (apx * abx + apy * aby) / ab_len_sq
    t = clamp(t, 0.0, 1.0)

    cx = ax + abx * t
    cy = ay + aby * t
    dx = px - cx
    dy = py - cy
    return dx * dx + dy * dy


# -----------------------------
# Player
# -----------------------------
class Player:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = PLAYER_X
        self.y = HEIGHT * 0.5
        self.vy = 0.0
        self.alive = True
        self.score = 0.0
        self.rect = pygame.Rect(0, 0, PLAYER_W, PLAYER_H)
        self._update_rect()

    def _update_rect(self):
        self.rect.center = (int(self.x), int(self.y))

    def update(self, dt, thrusting):
        if not self.alive:
            self.vy += GRAVITY * dt
            self.y += self.vy * dt
            self._update_rect()
            return

        ay = GRAVITY
        if thrusting:
            ay -= THRUST

        self.vy += ay * dt
        self.vy = clamp(self.vy, -MAX_RISE_SPEED, MAX_FALL_SPEED)
        self.y += self.vy * dt

        top_limit = PLAY_TOP + PLAYER_H / 2
        bottom_limit = PLAY_BOTTOM - PLAYER_H / 2

        if self.y < top_limit:
            self.y = top_limit
            if self.vy < 0:
                self.vy = 0
        elif self.y > bottom_limit:
            self.y = bottom_limit
            if self.vy > 0:
                self.vy = 0

        self._update_rect()
        self.score += dt * 10.0

    def draw(self, surface):
        body = pygame.Rect(0, 0, PLAYER_W, PLAYER_H)
        body.center = self.rect.center
        pygame.draw.rect(surface, (245, 210, 70), body, border_radius=8)

        helmet = pygame.Rect(body.x + 6, body.y + 8, body.w - 12, 14)
        pygame.draw.rect(surface, (255, 245, 220), helmet, border_radius=7)


# -----------------------------
# Zapper (accurate collision)
# -----------------------------
class Zapper:
    def __init__(self, x, y, length):
        self.x = float(x)
        self.y = float(y)
        self.length = float(length)
        self.phase = random.random() * 6.28

        r = random.random()
        if r < 0.55:
            self.angle_deg = 0
        elif r < 0.8:
            self.angle_deg = 90
        else:
            if random.random() < 0.5:
                self.angle_deg = random.uniform(30, 60)
            else:
                self.angle_deg = -random.uniform(30, 60)

    def endpoints(self):
        theta = math.radians(self.angle_deg)
        dx = math.cos(theta) * (self.length * 0.5)
        dy = math.sin(theta) * (self.length * 0.5)
        return self.x - dx, self.y - dy, self.x + dx, self.y + dy

    def update(self, dt, world_speed):
        self.x -= world_speed * dt
        self.phase += dt * 6.0

    def offscreen(self):
        ax, ay, bx, by = self.endpoints()
        return max(ax, bx) < -60

    def collides_player(self, player_rect):
        px, py = player_rect.center
        player_radius = min(player_rect.w, player_rect.h) * 0.45

        ax, ay, bx, by = self.endpoints()
        d2 = dist_point_to_segment_sq(px, py, ax, ay, bx, by)

        zapper_radius = ZAPPER_THICKNESS * 0.5
        hit_r = zapper_radius + player_radius
        return d2 <= hit_r * hit_r

    def draw(self, surface):
        base_color = (70, 160, 255)
        squiggle_color = (220, 245, 255)

        theta_draw = -self.angle_deg
        w = int(self.length)
        h = int(ZAPPER_THICKNESS)

        surf = pygame.Surface((w + 8, h + 8), pygame.SRCALPHA)
        rect_local = pygame.Rect(4, 4, w, h)
        pygame.draw.rect(surf, base_color, rect_local, border_radius=6)

        points = []
        steps = 14
        for i in range(steps + 1):
            t = i / steps
            px = 4 + t * w
            py = 4 + h / 2 + math.sin(self.phase + t * 10.0) * 5
            points.append((px, py))
        pygame.draw.lines(surf, squiggle_color, False, points, 2)

        rotated = pygame.transform.rotate(surf, theta_draw)
        rr = rotated.get_rect(center=(int(self.x), int(self.y)))
        surface.blit(rotated, rr.topleft)


# -----------------------------
# Missile + Warning
# -----------------------------
class MissileWarning:
    def __init__(self, y):
        self.y = y
        self.t = 0

    def update(self, dt):
        self.t += dt

    def done(self):
        return self.t >= MISSILE_WARNING_TIME

    def draw(self, surface):
        alpha = int(160 + 80 * math.sin(self.t * 18))
        warn_surf = pygame.Surface((WIDTH, 28), pygame.SRCALPHA)
        pygame.draw.rect(
            warn_surf,
            (255, 80, 80, alpha),
            pygame.Rect(WIDTH - 220, 0, 210, 28),
            border_radius=10,
        )
        surface.blit(warn_surf, (0, int(self.y - 14)))


class Missile:
    def __init__(self, y):
        self.x = WIDTH + 30
        self.y = y

    @property
    def rect(self):
        return pygame.Rect(
            int(self.x - MISSILE_W / 2),
            int(self.y - MISSILE_H / 2),
            MISSILE_W,
            MISSILE_H,
        )

    def update(self, dt):
        self.x -= MISSILE_SPEED * dt

    def offscreen(self):
        return self.x < -60

    def draw(self, surface):
        pygame.draw.rect(surface, (255, 80, 80), self.rect, border_radius=6)


# -----------------------------
# Particle
# -----------------------------
class Particle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = random.uniform(-140, -50)
        self.vy = random.uniform(-40, 40)
        self.life = random.uniform(0.25, 0.45)
        self.max_life = self.life
        self.size = random.uniform(3, 6)

    def update(self, dt):
        self.life -= dt
        self.x += self.vx * dt
        self.y += self.vy * dt

    def dead(self):
        return self.life <= 0

    def draw(self, surface):
        a = clamp(self.life / self.max_life, 0.0, 1.0)
        radius = max(1, int(self.size * a))
        s = pygame.Surface((radius * 2 + 2, radius * 2 + 2), pygame.SRCALPHA)
        pygame.draw.circle(
            s, (255, 200, 80, int(220 * a)), (radius + 1, radius + 1), radius
        )
        surface.blit(s, (self.x - radius, self.y - radius))


# -----------------------------
# Main Game
# -----------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Jetpack Rectangle")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("consolas", 20)
    big_font = pygame.font.SysFont("consolas", 34, bold=True)

    player = Player()
    zappers = []
    warnings = []
    missiles = []
    particles = []

    zapper_timer = random.uniform(*ZAPPER_SPAWN_EVERY)
    missile_timer = random.uniform(*MISSILE_SPAWN_EVERY)

    world_speed = SCROLL_SPEED

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        dt = min(dt, 1 / 30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r and not player.alive:
                    player.reset()
                    zappers.clear()
                    warnings.clear()
                    missiles.clear()
                    particles.clear()
                    world_speed = SCROLL_SPEED

        keys = pygame.key.get_pressed()
        thrusting = player.alive and (
            keys[pygame.K_SPACE] or keys[pygame.K_UP] or keys[pygame.K_w]
        )

        world_speed += SPEED_RAMP * dt
        player.update(dt, thrusting)

        if thrusting and player.alive:
            for _ in range(int(PARTICLE_RATE * dt)):
                particles.append(
                    Particle(player.rect.left + 6, player.rect.bottom - 10)
                )

        for p in particles:
            p.update(dt)
        particles = [p for p in particles if not p.dead()]

        if player.alive:
            zapper_timer -= dt
            if zapper_timer <= 0:
                zappers.append(
                    Zapper(
                        WIDTH + 40,
                        random.randint(PLAY_TOP + 40, PLAY_BOTTOM - 40),
                        random.randint(ZAPPER_MIN_LEN, ZAPPER_MAX_LEN),
                    )
                )
                zapper_timer = random.uniform(*ZAPPER_SPAWN_EVERY)

        for z in zappers:
            z.update(dt, world_speed)
        zappers = [z for z in zappers if not z.offscreen()]

        if player.alive:
            missile_timer -= dt
            if missile_timer <= 0:
                y = clamp(
                    player.y + random.uniform(-90, 90),
                    PLAY_TOP + 30,
                    PLAY_BOTTOM - 30,
                )
                warnings.append(MissileWarning(y))
                missile_timer = random.uniform(*MISSILE_SPAWN_EVERY)

        new_warnings = []
        for w in warnings:
            w.update(dt)
            if w.done():
                missiles.append(Missile(w.y))
            else:
                new_warnings.append(w)
        warnings = new_warnings

        for m in missiles:
            m.update(dt)
        missiles = [m for m in missiles if not m.offscreen()]

        if player.alive:
            for z in zappers:
                if z.collides_player(player.rect):
                    player.alive = False
                    break

        if player.alive:
            for m in missiles:
                if rects_collide(player.rect, m.rect):
                    player.alive = False
                    break

        screen.fill(BG_COLOR)

        for z in zappers:
            z.draw(screen)
        for w in warnings:
            w.draw(screen)
        for m in missiles:
            m.draw(screen)
        for p in particles:
            p.draw(screen)

        player.draw(screen)

        draw_text(screen, f"Score: {int(player.score):,}", (16, 12), font)
        draw_text(screen, f"Speed: {int(world_speed)}", (16, 36), font)

        if not player.alive:
            draw_text(screen, "CRASH!", (WIDTH // 2 - 70, HEIGHT // 2 - 60), big_font)

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()