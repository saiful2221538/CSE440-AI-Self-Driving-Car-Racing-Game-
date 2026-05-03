# 🏎️ Self-Driving Car Racing Game Using Reinforcement Learning

> A **Double Deep Q-Network (DDQN)** agent that learns to drive autonomously around a 2-D racing circuit — trained entirely from scratch through trial and error, with no hand-crafted driving logic.

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Hyperparameters](#-hyperparameters)
- [Results](#-results)
- [Known Issues Fixed](#-known-issues-fixed)
- [Future Work](#-future-work)
- [References](#-references)

---

## 🧠 Overview

This project implements a **self-driving car** inside a custom [pygame](https://www.pygame.org) racing simulator.
The car agent perceives the world through **18 simulated LIDAR rays** and a **velocity reading**, giving a 19-dimensional state vector.
A **DDQN** neural network maps states to Q-values over **9 discrete actions** (idle, forward, brake, steer, and diagonal combinations).

The agent is trained using:
- **Experience replay** — a circular buffer of 25,000 transitions
- **Target network** — a periodically synced copy that stabilises training
- **ε-greedy exploration** — decaying from 1.0 → 0.10 over training
- **Huber loss** — robust to large TD-error outliers early in training

---

## ✨ Features

- 🤖 **Self-Drive Mode** — watch the AI train itself in real time; speed up to 10×
- 🏁 **Race Mode** — compete against the trained AI with your arrow keys
- 🏆 **Leaderboard** — best lap times saved to `leaderboard.json`, ranked by time
- 📊 **Live HUD** — Points, Speed, Generation, ε, Loss, and action indicator boxes
- 💾 **Auto-save** — model weights saved every 10 episodes and on exit
- ⚡ **CPU-only optimised** — runs smoothly on laptops with no GPU

---

## ⚙️ How It Works

### State Space

The car casts 18 rays at fixed angular offsets from its heading. Each ray reports a normalised distance `(1000 - d) / 1000` to the nearest wall (1 = wall right there, 0 = clear). A normalised speed value is appended:

```
s ∈ ℝ¹⁹  =  [LIDAR₁, LIDAR₂, ..., LIDAR₁₈, v / v_max]
```

### Action Space

| ID | Action          | ID | Action           |
|:--:|:----------------|:--:|:-----------------|
| 0  | Idle            | 5  | Brake + Right    |
| 1  | Forward         | 6  | Brake + Left     |
| 2  | Turn Left       | 7  | Forward + Left   |
| 3  | Turn Right      | 8  | Forward + Right  |
| 4  | Brake           |    |                  |

### Reward Function

```
r = +1   →  car crosses the active checkpoint gate
r =  0   →  per-step living reward (no event)
r = -1   →  car collides with a wall  (episode ends)
```

### DDQN Update Rule

```
y  =  r  +  γ · Q_target(s', argmax_a Q_eval(s', a)) · (1 - done)

L(θ)  =  Huber( y  -  Q_eval(s, a; θ) )
```

The **eval network** selects the best action; the **target network** evaluates its value.
Decoupling selection from evaluation reduces overestimation bias compared to standard DQN.

---

## 📁 Project Structure

```
.
├── main.py                  # Headless training loop
├── game.py                  # Interactive Game UI (menu, race, leaderboard)
├── dqn.py                   # ReplayBuffer, Brain (MLP), DDQNAgent
├── environment.py           # RacingEnv, Car physics, LIDAR, collision
├── Walls.py                 # 48 line-segment track boundaries
├── Goals.py                 # 37 invisible checkpoint gates
├── track.png                # Track background image
├── car.png                  # Car sprite
├── requirements.txt         # Python dependencies
├── leaderboard.json         # Saved lap times (auto-generated)
├── ddqn_weights_eval*       # Saved eval network weights (auto-generated)
└── ddqn_weights_target*     # Saved target network weights (auto-generated)
```

---

## 🛠️ Installation

### Prerequisites

- Python 3.9 or higher
- No GPU required — CPU-only by default

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/self-driving-car-rl.git
cd self-driving-car-rl

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### `requirements.txt`

```
tensorflow>=2.10
pygame>=2.1
numpy>=1.23
```

> **Note:** TensorFlow is forced to CPU automatically via environment variables set in `dqn.py`. No CUDA setup is needed.

---

## 🚀 Usage

### Option A — Interactive Game UI *(recommended)*

```bash
python game.py
```

Opens the main menu with three modes:

| Mode | Description |
|------|-------------|
| **Self Drive (Train AI)** | Train the agent while watching it drive live |
| **Watch Trained AI** | Observe the trained model at 1× speed without training |
| **Race vs AI** | Play against the AI using your arrow keys |

Use the **Learning Rate** and **Training Speed** sliders before launching a mode.

---

### Option B — Headless Training *(faster)*

```bash
python main.py
```

Trains for 10,000 episodes without rendering. Model weights are saved every 10 episodes automatically.

---

### Controls

**Race Mode**

| Key | Action |
|:---:|--------|
| ↑ | Accelerate |
| ↓ | Brake / Reverse |
| ← | Steer Left |
| → | Steer Right |
| R | Restart race |
| ESC | Return to menu |

**Self-Drive Mode**

| Key | Action |
|:---:|--------|
| ↑ / ↓ | Increase / decrease training speed (1×–10×) |
| S | Manually save model now |
| R | Reset current episode |
| ESC | Return to menu |

---

## 📐 Hyperparameters

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Learning rate | η | 5 × 10⁻⁴ |
| Discount factor | γ | 0.99 |
| Initial exploration | ε₀ | 1.00 |
| Minimum exploration | ε_min | 0.10 |
| Exploration decay | δ_ε | 0.9995 |
| Batch size | B | 256 |
| Replay buffer size | \|B\| | 25,000 |
| Target sync interval | N_sync | 50 steps |
| Hidden layer width | — | 128 neurons × 2 layers |
| Max steps per episode | — | 1,000 |
| Stuck timeout | — | 100 steps with zero reward |

---

## 📊 Results

The agent achieves stable improvement across ~3,000 episodes on a CPU-only machine:

| Phase | Episodes | Avg. Reward | Observed Behaviour |
|-------|:--------:|:-----------:|-------------------|
| Random exploration | 0 – 200 | ~0 | Crashes within 50 steps |
| Rapid improvement | 200 – 800 | ~6 | Navigates early checkpoints |
| Policy refinement | 800 – 3,000 | ~20 | Smooth cornering, ~2 full laps |

By convergence (~episode 2,200), the agent consistently crosses **~22 checkpoints per episode** — roughly two full laps before a wall collision.

---

## 🐛 Known Issues Fixed

The original codebase had several critical bugs. All have been corrected in this version:

| Bug | Original Code | Fix Applied |
|-----|:-------------:|:-----------:|
| Output activation | `softmax` — forces outputs to sum to 1, destroys Q-value meaning | `linear` — correct for unbounded Q-values |
| Action space size | `n_actions=5` — agent never learned actions 5–8 | `n_actions=9` |
| Terminal state return | `None` — caused `np.array(None)` crash | Zero-vector `ℝ¹⁹` |
| Prediction verbosity | Progress bar printed every step — massive I/O slowdown | `verbose=0` |
| Loss function | MSE — sensitive to early outliers | Huber loss |
| Car image loading | Called before `pygame.init()` — crash on import | Lazy load after display init |

---

## 🔮 Future Work

- [ ] Continuous action space via **PPO** or **SAC** for finer steering control
- [ ] Recurrent architecture (**DRQN**) to handle partial observability beyond LIDAR range
- [ ] Procedural track generation for multi-track generalisation
- [ ] Multi-agent training — cars racing each other to produce emergent strategies
- [ ] Curriculum learning — start on wide straight tracks, graduate to tight circuits

---

## 📚 References

1. V. Mnih et al., *"Human-level control through deep reinforcement learning"*, **Nature**, 2015.
2. H. van Hasselt, A. Guez, D. Silver, *"Deep reinforcement learning with double Q-learning"*, **AAAI**, 2016.
3. R. S. Sutton & A. G. Barto, *Reinforcement Learning: An Introduction*, MIT Press, 2018.
4. D. P. Kingma & J. Ba, *"Adam: A method for stochastic optimization"*, **ICLR**, 2015.
5. Pygame — https://www.pygame.org

---

## 👤 Author

**Saiful Islam** `2221538642`  
Department of Computer Science & Engineering 
North South University, Dhaka, Bangladesh  
*CSE440 · Section 01 · Group 10*

---

## 📄 License

This project is released under the [MIT License](LICENSE).
