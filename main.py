# main.py  —  Training Loop
#
# Deep Q-Learning as described in lecture notes (pages 13-14):
#
#   Monte Carlo baseline:  estimate Q by averaging episode returns
#   Bootstrapping (SARSA): Q(s,a) <- (1-η)*Q(s,a) + η*[r + γ*Q(s',a')]
#   Q-learning:            Q_opt  <- (1-η)*Q_opt(s,a) + η*[r + γ*max_{a'} Q_opt(s',a')]
#   Deep RL:               Use neural net to estimate Q_opt  ← this file
#
# Key fix: n_actions = 9  (actions 0-8 as defined in Car.action())
#          input_dims = 19 (18 sensors + 1 velocity)

import pygame
import numpy as np

import environment
from dqn import DDQNAgent

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_EPISODES     = 10_000
MAX_STEPS      = 1_000     # max ticks per episode before forcing reset
RENDER_EVERY   = 10        # render every N episodes (set to 1 to always render)
SAVE_EVERY     = 10
REPLACE_EVERY  = 50        # sync target ← eval every N episodes

# Car has 9 discrete actions (0 = idle, 1-8 = movement combos)
# CRITICAL: was wrongly set to 5 in original code
N_ACTIONS  = environment.N_ACTIONS    # 9
INPUT_DIMS = environment.N_SENSORS + 1  # 19

# No reward for N consecutive idle ticks → assume stuck → end episode
STUCK_LIMIT = 100

# ── Agent ─────────────────────────────────────────────────────────────────────
agent = DDQNAgent(
    alpha          = 0.0005,      # η — learning rate
    gamma          = 0.99,        # γ — discount factor
    n_actions      = N_ACTIONS,
    epsilon        = 1.00,        # start fully exploratory
    epsilon_dec    = 0.9995,
    epsilon_end    = 0.10,
    batch_size     = 256,         # reduced from 512 for CPU speed
    input_dims     = INPUT_DIMS,
    mem_size       = 25_000,
    replace_target = REPLACE_EVERY,
    fname          = "ddqn_weights",
)

# Uncomment to resume training from a saved model:
# agent.load_model()

# ── Environment ───────────────────────────────────────────────────────────────
game = environment.RacingEnv()
game.fps = 60

# ── Tracking ──────────────────────────────────────────────────────────────────
scores      = []
eps_history = []


def run():
    for episode in range(N_EPISODES):

        game.reset()

        # Get initial observation (step with idle action to get first state)
        observation, _, _ = game.step(0)
        observation = np.array(observation, dtype=np.float32)

        score       = 0.0
        stuck_cnt   = 0
        done        = False
        render      = (episode % RENDER_EVERY == 0)

        step = 0
        while not done and step < MAX_STEPS:

            # Handle quit event even during training
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    agent.save_model()
                    game.close()
                    return

            # ── Choose action (ε-greedy) ─────────────────────────────────────
            action = agent.choose_action(observation)

            # ── Step environment ─────────────────────────────────────────────
            new_obs, reward, done = game.step(action)
            new_obs = np.array(new_obs, dtype=np.float32)

            # ── Stuck detection ──────────────────────────────────────────────
            # If the agent collects no reward for STUCK_LIMIT steps, end episode
            if reward == 0:
                stuck_cnt += 1
                if stuck_cnt > STUCK_LIMIT:
                    done = True
            else:
                stuck_cnt = 0

            score += reward

            # ── Store & learn ────────────────────────────────────────────────
            agent.remember(observation, action, reward, new_obs, done)
            agent.learn()

            observation = new_obs
            step += 1

            # ── Render ───────────────────────────────────────────────────────
            if render:
                game.render(action)

        # ── Episode end ───────────────────────────────────────────────────────
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg = np.mean(scores[max(0, episode - 100):episode + 1])

        if episode % SAVE_EVERY == 0 and episode > 0:
            agent.save_model()

        print(
            f"EP {episode:5d} | "
            f"Score: {score:7.2f} | "
            f"Avg(100): {avg:7.2f} | "
            f"ε: {agent.epsilon:.4f} | "
            f"Mem: {min(agent.memory.mem_cntr, agent.memory.mem_size):6d} | "
            f"Steps: {step}"
        )

    # Final save
    agent.save_model()
    game.close()
    print("Training complete.")


if __name__ == "__main__":
    run()