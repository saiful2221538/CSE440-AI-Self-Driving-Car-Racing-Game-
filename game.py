# game.py  —  Full Game UI
# Fixed to work with the actual environment.py and dqn.py
#
# Modes:
#   MainMenu         — launch hub with sliders
#   SelfDriveMode    — AI trains itself in real-time (watch it learn)
#   RaceMode         — Player (arrow keys) vs AI ghost car
#   LeaderboardScreen

import os
import json
import math
import time
import numpy as np
import pygame

# ── Local imports ─────────────────────────────────────────────────────────────
from environment import RacingEnv, Car, N_ACTIONS, N_SENSORS
from dqn import DDQNAgent

# ── Constants ─────────────────────────────────────────────────────────────────
SCREEN_W   = 1000
SCREEN_H   = 600
FPS        = 60
MODEL_PATH = "ddqn_weights"
LB_FILE    = "leaderboard.json"

# Colours
C_BG    = (0,   0,   0)
C_HUD   = (0,   255, 0)       # bright green — matches screenshot
C_HUD2  = (120, 120, 120)
C_BORDER= (255, 165, 0)       # orange track border

INPUT_DIMS = N_SENSORS + 1    # 19


# ─────────────────────────────────────────────────────────────────────────────
#  Leaderboard helpers
# ─────────────────────────────────────────────────────────────────────────────

def lb_load():
    if os.path.exists(LB_FILE):
        with open(LB_FILE) as f:
            return json.load(f)
    return []

def lb_save(entries):
    entries.sort(key=lambda e: e["time"])
    with open(LB_FILE, "w") as f:
        json.dump(entries[:10], f, indent=2)

def lb_add(name, lap_time):
    e = lb_load()
    e.append({"name": name, "time": round(lap_time, 3)})
    lb_save(e)


# ─────────────────────────────────────────────────────────────────────────────
#  Lap Tracker
# ─────────────────────────────────────────────────────────────────────────────

class LapTracker:
    """Detects when a car passes the start/finish gate."""
    GATE_X    = 50     # same as Car start x
    GATE_Y    = 300    # same as Car start y
    GATE_HALF = 55

    def __init__(self):
        self.reset()

    def reset(self):
        self.laps       = 0
        self.gate_left  = False
        self.lap_times  = []
        self._lap_start = time.time()

    def update(self, car):
        in_gate = (abs(car.x - self.GATE_X) < 20 and
                   abs(car.y - self.GATE_Y) < self.GATE_HALF)
        if not in_gate:
            self.gate_left = True
        elif self.gate_left:
            now = time.time()
            self.lap_times.append(now - self._lap_start)
            self._lap_start = now
            self.laps += 1
            self.gate_left = False

    @property
    def best(self):
        return min(self.lap_times) if self.lap_times else 0.0

    @property
    def elapsed(self):
        return time.time() - self._lap_start


# ─────────────────────────────────────────────────────────────────────────────
#  UI Widgets
# ─────────────────────────────────────────────────────────────────────────────

class Button:
    def __init__(self, x, y, w, h, text, color=(50, 50, 50)):
        self.rect  = pygame.Rect(x, y, w, h)
        self.text  = text
        self.color = color
        self._hov  = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self._hov = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            return True
        return False

    def draw(self, surf, font):
        c = tuple(min(255, v + 35) for v in self.color) if self._hov else self.color
        pygame.draw.rect(surf, c,           self.rect, border_radius=6)
        pygame.draw.rect(surf, (180,180,180), self.rect, 1, border_radius=6)
        s = font.render(self.text, True, (255, 255, 255))
        surf.blit(s, s.get_rect(center=self.rect.center))


class Slider:
    def __init__(self, x, y, w, lo, hi, val, label, fmt="{:.5f}"):
        self.rect  = pygame.Rect(x, y, w, 8)
        self.lo, self.hi = lo, hi
        self.val   = float(val)
        self.label = label
        self.fmt   = fmt
        self._drag = False

    @property
    def knob_x(self):
        t = (self.val - self.lo) / (self.hi - self.lo)
        return int(self.rect.left + t * self.rect.width)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if math.hypot(event.pos[0] - self.knob_x,
                          event.pos[1] - self.rect.centery) < 13:
                self._drag = True
        if event.type == pygame.MOUSEBUTTONUP:
            self._drag = False
        if event.type == pygame.MOUSEMOTION and self._drag:
            t = (event.pos[0] - self.rect.left) / max(1, self.rect.width)
            self.val = self.lo + max(0.0, min(1.0, t)) * (self.hi - self.lo)

    def draw(self, surf, font):
        pygame.draw.rect(surf, (40, 40, 40), self.rect, border_radius=4)
        fill = self.rect.copy()
        fill.width = max(0, self.knob_x - self.rect.left)
        pygame.draw.rect(surf, (0, 180, 80), fill, border_radius=4)
        pygame.draw.circle(surf, (200, 200, 200), (self.knob_x, self.rect.centery), 11)
        pygame.draw.circle(surf, (0, 220, 80),    (self.knob_x, self.rect.centery),  7)
        txt = f"{self.label}: {self.fmt.format(self.val)}"
        surf.blit(font.render(txt, True, (180, 180, 180)),
                  (self.rect.left, self.rect.top - 20))


# ─────────────────────────────────────────────────────────────────────────────
#  Self Drive Mode  (AI trains in real-time — matches screenshot style)
# ─────────────────────────────────────────────────────────────────────────────

class SelfDriveMode:
    STUCK_LIMIT  = 100
    MAX_STEPS    = 1000
    TRAIN_EVERY  = 4

    # Action label grid shown in HUD
    ACTION_LABELS = ["idle", "fwd", "L", "R", "bk", "bk+R", "bk+L", "fwd+L", "fwd+R"]

    def __init__(self, screen, clock, lr=5e-4, fast=1):
        self.screen = screen
        self.clock  = clock
        self.fast   = max(1, int(fast))

        self.font_l = pygame.font.SysFont("monospace", 28, bold=True)
        self.font_s = pygame.font.SysFont("monospace", 13)

        # Reuse existing env window by patching its screen reference
        self._env = RacingEnv.__new__(RacingEnv)
        self._env.screen     = screen
        self._env.width      = SCREEN_W
        self._env.height     = SCREEN_H
        self._env.fps        = FPS
        self._env.font       = pygame.font.Font(pygame.font.get_default_font(), 36)
        self._env.clock      = clock
        self._env.back_image = pygame.image.load("track.png").convert()
        self._env.back_rect  = self._env.back_image.get_rect()
        from Walls import getWalls
        from Goals import getGoals
        self._env.walls  = getWalls()
        self._env.goals  = getGoals()
        self._env.car    = Car(50, 300)
        self._env.game_reward = 0

        self.agent = DDQNAgent(
            alpha=lr, gamma=0.99,
            n_actions=N_ACTIONS, epsilon=1.0,
            epsilon_dec=0.9995, epsilon_end=0.10,
            batch_size=256, input_dims=INPUT_DIMS,
            mem_size=25_000, replace_target=50,
            fname=MODEL_PATH,
        )
        self.agent.load_model()

        self.points      = 0
        self.best_points = 0.0
        self.generation  = 0
        self.total_steps = 0
        self._action     = 0
        self._ep_reward  = 0.0

        obs, _, _ = self._env.step(0)
        self._obs = np.array(obs, dtype=np.float32)

    def run(self):
        running   = True
        stuck_cnt = 0
        step_cnt  = 0

        while running:
            self.clock.tick(FPS * self.fast)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: running = False
                    if event.key == pygame.K_s:      self.agent.save_model()
                    if event.key == pygame.K_r:
                        self._reset_episode()
                        stuck_cnt = step_cnt = 0
                    if event.key == pygame.K_UP:
                        self.fast = min(10, self.fast + 1)
                    if event.key == pygame.K_DOWN:
                        self.fast = max(1, self.fast - 1)

            for _ in range(self.fast):
                self._action = self.agent.choose_action(self._obs)
                next_obs, reward, done = self._env.step(self._action)
                next_obs = np.array(next_obs, dtype=np.float32)

                self.agent.remember(self._obs, self._action, reward, next_obs, done)
                self._obs        = next_obs
                self._ep_reward += reward
                self.total_steps += 1
                step_cnt         += 1
                self.points       = self._env.car.points

                if reward == 0:
                    stuck_cnt += 1
                else:
                    stuck_cnt = 0

                if self.total_steps % self.TRAIN_EVERY == 0:
                    self.agent.learn()

                if done or stuck_cnt > self.STUCK_LIMIT or step_cnt > self.MAX_STEPS:
                    self.generation += 1
                    if self._ep_reward > self.best_points:
                        self.best_points = self._ep_reward
                        self.agent.save_model()
                    self._reset_episode()
                    stuck_cnt = step_cnt = 0

            self._draw()

        self.agent.save_model()

    def _reset_episode(self):
        self._env.reset()
        obs, _, _ = self._env.step(0)
        self._obs       = np.array(obs, dtype=np.float32)
        self._ep_reward = 0.0
        self.points     = 0

    def _draw(self):
        self.screen.blit(self._env.back_image, self._env.back_rect)
        self._env.car._ensure_image()
        self._env.car.draw(self.screen)
        self._draw_rays()
        self._draw_hud()
        pygame.display.flip()

    def _draw_rays(self):
        for pt in self._env.car.closestRays:
            pygame.draw.line(self.screen, (0, 200, 100),
                             (int(self._env.car.x), int(self._env.car.y)),
                             (int(pt.x), int(pt.y)), 1)

    def _draw_hud(self):
        # Top-left: Points
        self.screen.blit(
            self.font_l.render(f"Points {int(self.points)}", True, C_HUD),
            (18, 14))

        # Top-right: Speed
        spd = int(abs(self._env.car.vel))
        self.screen.blit(
            self.font_l.render(f"Speed {spd}", True, C_HUD),
            (SCREEN_W - 200, 14))

        # Action indicator boxes (3 columns × 3 rows grid)
        bx = SCREEN_W - 18 - 3 * 44
        by = 54
        box_sz, gap = 36, 6
        for i, lbl in enumerate(self.ACTION_LABELS):
            rx = bx + (i % 3) * (box_sz + gap)
            ry = by + (i // 3) * (box_sz + gap)
            if i == self._action:
                pygame.draw.rect(self.screen, C_HUD, (rx, ry, box_sz, box_sz))
                s = self.font_s.render(lbl, True, (0, 0, 0))
            else:
                pygame.draw.rect(self.screen, C_BG,       (rx, ry, box_sz, box_sz))
                pygame.draw.rect(self.screen, (180,180,180),(rx, ry, box_sz, box_sz), 1)
                s = self.font_s.render(lbl, True, (180, 180, 180))
            self.screen.blit(s, s.get_rect(center=(rx + box_sz//2, ry + box_sz//2)))

        # Bottom info strip
        strip = [
            f"Gen: {self.generation}",
            f"Best: {int(self.best_points)}",
            f"ε: {self.agent.epsilon:.3f}",
            f"Steps: {self.total_steps}",
            f"Speed x{self.fast}",
        ]
        for i, t in enumerate(strip):
            self.screen.blit(
                self.font_s.render(t, True, C_HUD2),
                (18 + i * 170, SCREEN_H - 26))

        hint = self.font_s.render(
            "S=save  R=reset  ↑↓=speed  ESC=menu", True, (60, 60, 60))
        self.screen.blit(hint, hint.get_rect(center=(SCREEN_W // 2, SCREEN_H - 26)))


# ─────────────────────────────────────────────────────────────────────────────
#  Race Mode  (Player arrow keys vs AI)
# ─────────────────────────────────────────────────────────────────────────────

class RaceMode:
    """
    Player controls car with arrow keys.
    AI uses the trained DDQN model.
    First to 3 laps wins.
    """
    LAP_GOAL = 3

    def __init__(self, screen, clock):
        self.screen = screen
        self.clock  = clock
        self.font_l = pygame.font.SysFont("monospace", 26, bold=True)
        self.font_s = pygame.font.SysFont("monospace", 13)

        self.back_image = pygame.image.load("track.png").convert()
        self.back_rect  = self.back_image.get_rect()

        # AI agent (load saved weights)
        self._agent = DDQNAgent(
            alpha=0.0005, gamma=0.99,
            n_actions=N_ACTIONS, epsilon=0.0,   # no exploration during race
            batch_size=256, input_dims=INPUT_DIMS,
            fname=MODEL_PATH,
        )
        self._agent.load_model()

        from Walls import getWalls
        from Goals import getGoals
        self._walls = getWalls()
        self._goals_template = getGoals   # callable to fresh copy

        self._reset()

    def _reset(self):
        from Goals import getGoals
        self.player_car = Car(50, 275)
        self.ai_car     = Car(50, 325)

        self._player_goals = getGoals()
        self._ai_goals     = getGoals()

        self._pl_tracker = LapTracker()
        self._ai_tracker = LapTracker()

        self._pl_crashed = False
        self._ai_crashed = False
        self._winner     = None
        self._race_done  = False

        # Bootstrap first AI observation
        self._ai_obs = self._get_ai_obs()

    def _get_ai_obs(self):
        obs = self.ai_car.cast(self._walls)
        return np.array(obs, dtype=np.float32)

    def run(self):
        running = True
        while running:
            self.clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: running = False
                    if event.key == pygame.K_r:      self._reset()

            if not self._race_done:
                self._update()

            self._draw()
            pygame.display.flip()

        if self._pl_tracker.best > 0:
            lb_add("Player", self._pl_tracker.best)
        if self._ai_tracker.best > 0:
            lb_add("AI",     self._ai_tracker.best)

    def _update(self):
        # ── Player controls ───────────────────────────────────────────────────
        if not self._pl_crashed:
            keys = pygame.key.get_pressed()
            if   keys[pygame.K_UP]    and keys[pygame.K_LEFT]:  self.player_car.action(7)
            elif keys[pygame.K_UP]    and keys[pygame.K_RIGHT]: self.player_car.action(8)
            elif keys[pygame.K_DOWN]  and keys[pygame.K_LEFT]:  self.player_car.action(6)
            elif keys[pygame.K_DOWN]  and keys[pygame.K_RIGHT]: self.player_car.action(5)
            elif keys[pygame.K_UP]:                              self.player_car.action(1)
            elif keys[pygame.K_DOWN]:                            self.player_car.action(4)
            elif keys[pygame.K_LEFT]:                            self.player_car.action(2)
            elif keys[pygame.K_RIGHT]:                           self.player_car.action(3)
            else:                                                self.player_car.action(0)

            self.player_car.update()

            for wall in self._walls:
                if self.player_car.collision(wall):
                    self._pl_crashed = True
                    break

        # ── Goal scoring (player) ─────────────────────────────────────────────
        n = len(self._player_goals)
        for idx, goal in enumerate(self._player_goals):
            if goal.isactiv and self.player_car.score(goal):
                goal.isactiv = False
                self._player_goals[(idx - 1) % n].isactiv = True
        self._pl_tracker.update(self.player_car)

        # ── AI step ───────────────────────────────────────────────────────────
        if not self._ai_crashed:
            action = self._agent.choose_action(self._ai_obs)
            self.ai_car.action(action)
            self.ai_car.update()
            self._ai_obs = self._get_ai_obs()

            for wall in self._walls:
                if self.ai_car.collision(wall):
                    self._ai_crashed = True
                    break

        # Goal scoring (AI)
        n = len(self._ai_goals)
        for idx, goal in enumerate(self._ai_goals):
            if goal.isactiv and self.ai_car.score(goal):
                goal.isactiv = False
                self._ai_goals[(idx - 1) % n].isactiv = True
        self._ai_tracker.update(self.ai_car)

        # ── Win check ─────────────────────────────────────────────────────────
        if self._pl_tracker.laps >= self.LAP_GOAL:
            self._winner    = "PLAYER"
            self._race_done = True
        elif self._ai_tracker.laps >= self.LAP_GOAL:
            self._winner    = "AI"
            self._race_done = True

    def _draw(self):
        self.screen.blit(self.back_image, self.back_rect)

        # AI rays
        for pt in self.ai_car.closestRays:
            pygame.draw.line(self.screen, (0, 150, 80),
                             (int(self.ai_car.x), int(self.ai_car.y)),
                             (int(pt.x), int(pt.y)), 1)

        self.player_car._ensure_image()
        self.ai_car._ensure_image()
        self.ai_car.draw(self.screen)
        self.player_car.draw(self.screen)

        self._draw_panels()
        if self._race_done:
            self._draw_winner()

    def _draw_panels(self):
        pl = self._pl_tracker
        ai = self._ai_tracker

        # Player panel — top left
        panel = pygame.Surface((190, 125), pygame.SRCALPHA)
        panel.fill((0, 0, 60, 180))
        self.screen.blit(panel, (8, 8))
        lines = [
            ("── YOU ──",                              (100, 180, 255)),
            (f"Lap  {pl.laps}/{self.LAP_GOAL}",        (200, 200, 255)),
            (f"Time {pl.elapsed:.2f}s",                (200, 200, 255)),
            (f"Best {pl.best:.2f}s",                   C_HUD),
            ("CRASHED!" if self._pl_crashed else "",   (255, 60,  60)),
        ]
        for i, (t, c) in enumerate(lines):
            if t:
                self.screen.blit(self.font_s.render(t, True, c), (14, 14 + i * 22))

        # AI panel — top right
        panel2 = pygame.Surface((190, 125), pygame.SRCALPHA)
        panel2.fill((60, 0, 0, 180))
        self.screen.blit(panel2, (SCREEN_W - 198, 8))
        lines2 = [
            ("── AI ──",                               (255, 160, 160)),
            (f"Lap  {ai.laps}/{self.LAP_GOAL}",        (255, 200, 200)),
            (f"Time {ai.elapsed:.2f}s",                (255, 200, 200)),
            (f"Best {ai.best:.2f}s",                   C_HUD),
            ("CRASHED!" if self._ai_crashed else "",   (255, 60,  60)),
        ]
        for i, (t, c) in enumerate(lines2):
            if t:
                self.screen.blit(self.font_s.render(t, True, c),
                                 (SCREEN_W - 192, 14 + i * 22))

        hint = self.font_s.render(
            "Arrows = drive   R = restart   ESC = back", True, (80, 80, 80))
        self.screen.blit(hint, hint.get_rect(center=(SCREEN_W // 2, SCREEN_H - 18)))

    def _draw_winner(self):
        ov = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        ov.fill((0, 0, 0, 130))
        self.screen.blit(ov, (0, 0))
        c = (100, 180, 255) if self._winner == "PLAYER" else (255, 100, 100)
        s = self.font_l.render(f"  {self._winner} WINS!  ", True, c)
        self.screen.blit(s, s.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 - 20)))
        s2 = self.font_s.render("R = race again   ESC = exit", True, (200, 200, 200))
        self.screen.blit(s2, s2.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 + 24)))


# ─────────────────────────────────────────────────────────────────────────────
#  Leaderboard Screen
# ─────────────────────────────────────────────────────────────────────────────

class LeaderboardScreen:
    def __init__(self, screen, clock):
        self.screen = screen
        self.clock  = clock
        self.font_l = pygame.font.SysFont("monospace", 28, bold=True)
        self.font_m = pygame.font.SysFont("monospace", 20)
        self.font_s = pygame.font.SysFont("monospace", 14)

    def run(self):
        entries = lb_load()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type in (pygame.QUIT, pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                    running = False
            self._draw(entries)
            pygame.display.flip()
            self.clock.tick(30)

    def _draw(self, entries):
        self.screen.fill(C_BG)
        t = self.font_l.render("  LEADERBOARD  ", True, C_HUD)
        self.screen.blit(t, t.get_rect(center=(SCREEN_W // 2, 55)))
        pygame.draw.line(self.screen, C_HUD,
                         (SCREEN_W//2 - 200, 82), (SCREEN_W//2 + 200, 82), 2)

        if not entries:
            m = self.font_m.render("No times yet — race first!", True, (120, 120, 120))
            self.screen.blit(m, m.get_rect(center=(SCREEN_W//2, SCREEN_H//2)))
        else:
            hdr = self.font_m.render(f"{'#':<4}{'Driver':<14}{'Best Lap':>9}", True, (180,160,60))
            self.screen.blit(hdr, hdr.get_rect(center=(SCREEN_W//2, 110)))
            medals = {0:"1st", 1:"2nd", 2:"3rd"}
            colors = {0:(255,215,0), 1:(200,200,200), 2:(205,127,50)}
            for i, e in enumerate(entries[:10]):
                rank = medals.get(i, f" {i+1}.")
                c    = colors.get(i, (160, 160, 160))
                row  = f"{rank:<4}{e['name']:<14}{e['time']:>7.3f}s"
                s    = self.font_m.render(row, True, c)
                self.screen.blit(s, s.get_rect(center=(SCREEN_W//2, 150 + i*34)))

        hint = self.font_s.render("Press any key …", True, (60, 60, 60))
        self.screen.blit(hint, hint.get_rect(center=(SCREEN_W//2, SCREEN_H - 28)))


# ─────────────────────────────────────────────────────────────────────────────
#  Main Menu
# ─────────────────────────────────────────────────────────────────────────────

class MainMenu:
    def __init__(self):
        if not pygame.get_init():
            pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("RL Racing — Deep Q-Network")
        self.clock  = pygame.time.Clock()

        self.font_xl = pygame.font.SysFont("monospace", 46, bold=True)
        self.font_l  = pygame.font.SysFont("monospace", 22, bold=True)
        self.font_m  = pygame.font.SysFont("monospace", 18)
        self.font_s  = pygame.font.SysFont("monospace", 13)

        self.back_image = pygame.image.load("track.png").convert()
        self.back_rect  = self.back_image.get_rect()

        # Sliders
        sx = SCREEN_W // 2 - 200
        self.sl_lr   = Slider(sx, 200, 400, 1e-5, 1e-3, 5e-4, "Learning Rate η")
        self.sl_fast = Slider(sx, 258, 400, 1, 10, 1,
                              "Training Speed x", "{:.0f}")

        # Buttons
        bw, bh = 320, 50
        cx = SCREEN_W // 2 - bw // 2
        self.buttons = {
            "train": Button(cx, 300, bw, bh, "  Self Drive (Train AI)",  (20, 100, 50)),
            "race":  Button(cx, 365, bw, bh, "  Race vs AI",             (100, 40, 160)),
            "board": Button(cx, 430, bw, bh, "  Leaderboard",            (130, 100, 20)),
            "quit":  Button(cx, 510, bw, bh, "  Quit",                   (140, 30, 30)),
        }

    def run(self):
        while True:
            self.clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                self.sl_lr.handle_event(event)
                self.sl_fast.handle_event(event)
                for key, btn in self.buttons.items():
                    if btn.handle_event(event):
                        if self._handle(key) == "quit":
                            pygame.quit()
                            return
            self._draw()

    def _handle(self, key):
        lr   = self.sl_lr.val
        fast = max(1, int(self.sl_fast.val))
        if key == "train":
            SelfDriveMode(self.screen, self.clock, lr=lr, fast=fast).run()
        elif key == "race":
            RaceMode(self.screen, self.clock).run()
        elif key == "board":
            LeaderboardScreen(self.screen, self.clock).run()
        elif key == "quit":
            return "quit"

    def _draw(self):
        # Dimmed track background
        self.screen.blit(self.back_image, self.back_rect)
        ov = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        ov.fill((0, 0, 0, 185))
        self.screen.blit(ov, (0, 0))

        # Title
        t1 = self.font_xl.render("RL RACING", True, C_HUD)
        self.screen.blit(t1, t1.get_rect(center=(SCREEN_W // 2, 75)))

        t2 = self.font_s.render(
            "Deep Q-Network  ·  Q-learning  ·  Self Driving Car  ·  Race Mode",
            True, (100, 100, 100))
        self.screen.blit(t2, t2.get_rect(center=(SCREEN_W // 2, 118)))

        # Model status
        ok  = os.path.exists(MODEL_PATH + "_eval.index")
        c   = C_HUD if ok else (200, 50, 50)
        msg = "  Model: READY  " if ok else "  Model: NOT TRAINED YET  "
        ms  = self.font_s.render(msg, True, c)
        self.screen.blit(ms, ms.get_rect(center=(SCREEN_W // 2, 148)))

        # Sliders + buttons
        self.sl_lr.draw(self.screen, self.font_s)
        self.sl_fast.draw(self.screen, self.font_s)
        for btn in self.buttons.values():
            btn.draw(self.screen, self.font_m)

        pygame.display.flip()


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    MainMenu().run()