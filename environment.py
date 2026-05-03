# environment.py  —  Racing Environment
# Fixes:
#   - pygame.image.load moved AFTER pygame.init() check
#   - step() returns zero-vector (not None) on terminal state
#   - render flag controls display so training can run headless/fast
#   - wall/goal imports kept identical to originals

import math
import pygame
import numpy as np

# ── Reward constants (tune these) ────────────────────────────────────────────
GOAL_REWARD  =  1
LIFE_REWARD  =  0       # small negative encourages speed; 0 = neutral
PENALTY      = -1


# ─────────────────────────────────────────────────────────────────────────────
#  Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

class myPoint:
    __slots__ = ("x", "y")
    def __init__(self, x, y):
        self.x = x
        self.y = y

class myLine:
    __slots__ = ("pt1", "pt2")
    def __init__(self, pt1, pt2):
        self.pt1 = myPoint(pt1.x, pt1.y)
        self.pt2 = myPoint(pt2.x, pt2.y)


def distance(pt1, pt2):
    return math.hypot(pt1.x - pt2.x, pt1.y - pt2.y)


def rotate(origin, point, angle):
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    dx, dy       = point.x - origin.x, point.y - origin.y
    return myPoint(
        origin.x + cos_a * dx - sin_a * dy,
        origin.y + sin_a * dx + cos_a * dy,
    )


def rotateRect(pt1, pt2, pt3, pt4, angle):
    cx = (pt1.x + pt3.x) / 2
    cy = (pt1.y + pt3.y) / 2
    center = myPoint(cx, cy)
    return (rotate(center, pt1, angle),
            rotate(center, pt2, angle),
            rotate(center, pt3, angle),
            rotate(center, pt4, angle))


# ─────────────────────────────────────────────────────────────────────────────
#  Ray  (for sensor / LIDAR casting)
# ─────────────────────────────────────────────────────────────────────────────

class Ray:
    RAY_LEN = 1000

    def __init__(self, x, y, angle):
        self.x     = x
        self.y     = y
        self.angle = angle

    def cast(self, wall):
        """Return intersection point with wall, or None."""
        x1, y1, x2, y2 = wall.x1, wall.y1, wall.x2, wall.y2

        vec = rotate(myPoint(0, 0), myPoint(0, -self.RAY_LEN), self.angle)
        x3, y3 = self.x, self.y
        x4, y4 = self.x + vec.x, self.y + vec.y

        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if den == 0:
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

        if 0 < t < 1 and 0 < u < 1:
            return myPoint(
                math.floor(x1 + t * (x2 - x1)),
                math.floor(y1 + t * (y2 - y1)),
            )
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Car
# ─────────────────────────────────────────────────────────────────────────────

# Number of sensor rays = 18, plus 1 velocity value = 19 total observations
N_SENSORS    = 18
N_ACTIONS    = 9     # actions 0–8 (see action() below)

class Car:
    MAX_VEL = 15

    def __init__(self, x, y):
        self.start_x = x
        self.start_y = y
        self.width   = 14
        self.height  = 30

        # Pygame image — loaded lazily so we don't need pygame at import time
        self._image_loaded    = False
        self.original_image   = None
        self.image            = None
        self.rect             = None

        self.points = 0
        self._reset_kinematics()

    # ── Lazy image load ───────────────────────────────────────────────────────

    def _ensure_image(self):
        if not self._image_loaded:
            try:
                self.original_image = pygame.image.load("car.png").convert()
                self.original_image.set_colorkey((0, 0, 0))
                self.image = self.original_image
                self.rect  = self.image.get_rect().move(self.x, self.y)
            except Exception:
                # Fallback: simple colored rectangle if car.png is missing
                self.original_image = pygame.Surface((self.width, self.height))
                self.original_image.fill((0, 200, 255))
                self.image = self.original_image
                self.rect  = self.image.get_rect().move(self.x, self.y)
            self._image_loaded = True

    # ── Kinematics reset ──────────────────────────────────────────────────────

    def _reset_kinematics(self):
        self.x     = self.start_x
        self.y     = self.start_y
        self.velX  = 0.0
        self.velY  = 0.0
        self.vel   = 0.0
        self.angle = math.radians(180)
        self.soll_angle = self.angle

        # Axis-aligned corners (updated each step)
        hw, hh = self.width / 2, self.height / 2
        base   = myPoint(self.start_x, self.start_y)
        self.pt1 = myPoint(base.x - hw, base.y - hh)
        self.pt2 = myPoint(base.x + hw, base.y - hh)
        self.pt3 = myPoint(base.x + hw, base.y + hh)
        self.pt4 = myPoint(base.x - hw, base.y + hh)

        # Rotated corners
        self.p1, self.p2, self.p3, self.p4 = self.pt1, self.pt2, self.pt3, self.pt4

        self.closestRays = []

    # ── Actions ───────────────────────────────────────────────────────────────
    # 0: idle   1: fwd   2: turn-L   3: turn-R
    # 4: brake  5: bk+R  6: bk+L    7: fwd+L   8: fwd+R

    def action(self, choice):
        d = 1  # dvel
        if choice == 1:
            self._accelerate(d)
        elif choice == 2:
            self._turn(-1)
        elif choice == 3:
            self._turn(1)
        elif choice == 4:
            self._accelerate(-d)
        elif choice == 5:
            self._accelerate(-d); self._turn(1)
        elif choice == 6:
            self._accelerate(-d); self._turn(-1)
        elif choice == 7:
            self._accelerate(d);  self._turn(-1)
        elif choice == 8:
            self._accelerate(d);  self._turn(1)
        # choice == 0: idle — no-op

    def _accelerate(self, dv):
        self.vel = max(-self.MAX_VEL,
                       min(self.MAX_VEL, self.vel + dv * 2))

    def _turn(self, direction):
        self.soll_angle += direction * math.radians(15)

    # ── Physics update ────────────────────────────────────────────────────────

    def update(self):
        self.angle = self.soll_angle

        vec = rotate(myPoint(0, 0), myPoint(0, self.vel), self.angle)
        self.velX, self.velY = vec.x, vec.y

        self.x += self.velX
        self.y += self.velY

        # Translate corners
        vx, vy = self.velX, self.velY
        self.pt1 = myPoint(self.pt1.x + vx, self.pt1.y + vy)
        self.pt2 = myPoint(self.pt2.x + vx, self.pt2.y + vy)
        self.pt3 = myPoint(self.pt3.x + vx, self.pt3.y + vy)
        self.pt4 = myPoint(self.pt4.x + vx, self.pt4.y + vy)

        # Rotate corners around center
        self.p1, self.p2, self.p3, self.p4 = rotateRect(
            self.pt1, self.pt2, self.pt3, self.pt4, self.soll_angle)

        # Update sprite
        if self._image_loaded:
            self.image = pygame.transform.rotate(
                self.original_image,
                90 - math.degrees(self.soll_angle),
            )
            cx, cy = self.rect.center
            self.rect = self.image.get_rect()
            self.rect.center = (cx, cy)

    # ── Sensor casting  (returns 19-element observation vector) ───────────────

    def cast(self, walls):
        angles = [
            self.soll_angle,
            self.soll_angle - math.radians(10),
            self.soll_angle + math.radians(10),
            self.soll_angle - math.radians(20),
            self.soll_angle + math.radians(20),
            self.soll_angle - math.radians(30),
            self.soll_angle + math.radians(30),
            self.soll_angle - math.radians(45),
            self.soll_angle + math.radians(45),
            self.soll_angle - math.radians(90),
            self.soll_angle + math.radians(90),
            self.soll_angle - math.radians(135),
            self.soll_angle + math.radians(135),
            self.soll_angle + math.radians(180),
            # Corner rays
            self.soll_angle + math.radians(90),   # from p1 (left-front)
            self.soll_angle - math.radians(90),   # from p2 (right-front)
            self.soll_angle,                       # from p1 forward
            self.soll_angle,                       # from p2 forward
        ]
        origins = (
            [(self.x, self.y)] * 14
            + [(self.p1.x, self.p1.y), (self.p2.x, self.p2.y),
               (self.p1.x, self.p1.y), (self.p2.x, self.p2.y)]
        )

        MAX_DIST   = 1000.0
        observations = []
        self.closestRays = []

        for (ox, oy), ang in zip(origins, angles):
            ray    = Ray(ox, oy, ang)
            record = MAX_DIST
            closest = None

            for wall in walls:
                pt = ray.cast(wall)
                if pt:
                    d = distance(myPoint(ox, oy), pt)
                    if d < record:
                        record  = d
                        closest = pt

            if closest:
                self.closestRays.append(closest)
            observations.append((MAX_DIST - record) / MAX_DIST)   # 1=close, 0=far

        # Append normalised speed
        observations.append(self.vel / self.MAX_VEL)
        return observations   # length = N_SENSORS + 1 = 19

    # ── Collision detection ───────────────────────────────────────────────────

    def collision(self, wall):
        car_lines = [
            myLine(self.p1, self.p2),
            myLine(self.p2, self.p3),
            myLine(self.p3, self.p4),
            myLine(self.p4, self.p1),
        ]
        x1, y1, x2, y2 = wall.x1, wall.y1, wall.x2, wall.y2

        for li in car_lines:
            x3, y3 = li.pt1.x, li.pt1.y
            x4, y4 = li.pt2.x, li.pt2.y

            den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if den == 0:
                continue

            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
            u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

            if 0 < t < 1 and 0 < u < 1:
                return True
        return False

    # ── Goal crossing ─────────────────────────────────────────────────────────

    def score(self, goal):
        """Returns True if car's front ray crosses the goal line."""
        vec  = rotate(myPoint(0, 0), myPoint(0, -50), self.angle)
        line = myLine(myPoint(self.x, self.y),
                      myPoint(self.x + vec.x, self.y + vec.y))

        x1, y1, x2, y2 = goal.x1, goal.y1, goal.x2, goal.y2
        x3, y3 = line.pt1.x, line.pt1.y
        x4, y4 = line.pt2.x, line.pt2.y

        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if den == 0:
            return False

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

        if 0 < t < 1 and 0 < u < 1:
            pt = (math.floor(x1 + t * (x2 - x1)),
                  math.floor(y1 + t * (y2 - y1)))
            if distance(myPoint(self.x, self.y),
                        myPoint(pt[0], pt[1])) < 20:
                self.points += GOAL_REWARD
                return True
        return False

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self):
        self.points = 0
        self._reset_kinematics()
        if self._image_loaded:
            self.rect.center = (self.x, self.y)

    # ── Draw ──────────────────────────────────────────────────────────────────

    def draw(self, win):
        self._ensure_image()
        win.blit(self.image, self.rect)


# ─────────────────────────────────────────────────────────────────────────────
#  Racing Environment
# ─────────────────────────────────────────────────────────────────────────────

class RacingEnv:
    """
    Gym-style environment wrapper.

    step() returns (new_state, reward, done)
    new_state is always a list of floats (never None) for safe np.array() wrapping.
    """

    def __init__(self):
        # Lazy pygame init so environment can be imported without a display
        if not pygame.get_init():
            pygame.init()

        self.fps    = 120
        self.width  = 1000
        self.height = 600

        self.screen     = pygame.display.set_mode((self.width, self.height))
        self.back_image = pygame.image.load("track.png").convert()
        self.back_rect  = self.back_image.get_rect()
        pygame.display.set_caption("RACING DQN")

        self.font = pygame.font.Font(pygame.font.get_default_font(), 36)
        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        from Walls import getWalls
        from Goals import getGoals

        self.car          = Car(50, 300)
        self.walls        = getWalls()
        self.goals        = getGoals()
        self.game_reward  = 0

    def step(self, action):
        done = False
        self.car.action(action)
        self.car.update()
        reward = LIFE_REWARD

        # ── Goal reward ───────────────────────────────────────────────────────
        n = len(self.goals)
        for idx, goal in enumerate(self.goals):
            if goal.isactiv and self.car.score(goal):
                goal.isactiv = False
                self.goals[(idx - 1) % n].isactiv = True
                reward += GOAL_REWARD

        # ── Collision penalty ─────────────────────────────────────────────────
        for wall in self.walls:
            if self.car.collision(wall):
                reward += PENALTY
                done = True
                break

        # ── Observation ───────────────────────────────────────────────────────
        new_state = self.car.cast(self.walls)
        # Return zero-vector on terminal state instead of None
        # (prevents np.array(None) crash in training loop)
        if done:
            new_state = [0.0] * (N_SENSORS + 1)

        return new_state, reward, done

    def render(self, action):
        DRAW_WALLS = False
        DRAW_GOALS = False
        DRAW_RAYS  = False

        self.screen.blit(self.back_image, self.back_rect)

        if DRAW_WALLS:
            for wall in self.walls:
                wall.draw(self.screen)
        if DRAW_GOALS:
            for goal in self.goals:
                goal.draw(self.screen)

        self.car._ensure_image()
        self.car.draw(self.screen)

        if DRAW_RAYS:
            for i, pt in enumerate(self.car.closestRays):
                pygame.draw.circle(self.screen, (0, 0, 255), (int(pt.x), int(pt.y)), 5)
                src = (self.car.x, self.car.y)
                pygame.draw.line(self.screen, (255, 255, 255), src, (int(pt.x), int(pt.y)), 1)

        # Action indicator boxes (matches original HUD)
        boxes = [(800,100), (850,100), (900,100), (850,50)]
        for bx, by in boxes:
            pygame.draw.rect(self.screen, (255, 255, 255), (bx, by, 40, 40), 2)

        action_fills = {
            4: [(850,50)],
            6: [(850,50),(800,100)],
            5: [(850,50),(900,100)],
            1: [(850,100)],
            8: [(850,100),(800,100)],
            7: [(850,100),(900,100)],
            2: [(800,100)],
            3: [(900,100)],
        }
        for bx, by in action_fills.get(action, []):
            pygame.draw.rect(self.screen, (0, 255, 0), (bx, by, 40, 40))

        # HUD
        pts_surf = self.font.render(f"Points {self.car.points}", True, pygame.Color("green"))
        spd_surf = self.font.render(f"Speed {int(abs(self.car.vel))}", True, pygame.Color("green"))
        self.screen.blit(pts_surf, (0,  0))
        self.screen.blit(spd_surf, (800, 0))

        self.clock.tick(self.fps)
        pygame.display.update()

    def close(self):
        pygame.quit()