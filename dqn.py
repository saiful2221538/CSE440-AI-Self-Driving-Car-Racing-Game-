# dqn.py  —  DDQN Agent

import os

# Force CPU before importing TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
tf.get_logger().setLevel("ERROR")


# ─────────────────────────────────────────────────────────────
# Replay Buffer
# ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros(
            (max_size, input_shape), dtype=np.float32
        )

        self.new_state_memory = np.zeros(
            (max_size, input_shape), dtype=np.float32
        )

        self.action_memory = np.zeros(
            (max_size, n_actions), dtype=np.int8
        )

        self.reward_memory = np.zeros(
            max_size, dtype=np.float32
        )

        self.terminal_memory = np.zeros(
            max_size, dtype=np.float32
        )

    def store(self, state, action, reward, state_, done):
        idx = self.mem_cntr % self.mem_size

        self.state_memory[idx] = state
        self.new_state_memory[idx] = state_

        one_hot = np.zeros(
            self.action_memory.shape[1],
            dtype=np.int8
        )

        one_hot[action] = 1
        self.action_memory[idx] = one_hot

        self.reward_memory[idx] = reward

        # 1 if not terminal, 0 if done
        self.terminal_memory[idx] = 1 - int(done)

        self.mem_cntr += 1

    def sample(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(
            max_mem,
            batch_size,
            replace=False
        )

        return (
            self.state_memory[batch],
            self.action_memory[batch],
            self.reward_memory[batch],
            self.new_state_memory[batch],
            self.terminal_memory[batch],
        )


# ─────────────────────────────────────────────────────────────
# Neural Network
# ─────────────────────────────────────────────────────────────

class Brain:

    def __init__(self, n_states, n_actions, lr=0.0005):
        self.n_states = n_states
        self.n_actions = n_actions
        self.model = self._build(lr)

    def _build(self, lr):

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.n_states,)),

            tf.keras.layers.Dense(
                128,
                activation="relu"
            ),

            tf.keras.layers.Dense(
                128,
                activation="relu"
            ),

            tf.keras.layers.Dense(
                self.n_actions,
                activation="linear"
            ),
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=lr
            ),
            loss=tf.keras.losses.Huber()
        )

        return model

    def predict(self, states):
        return self.model.predict(
            states,
            verbose=0
        )

    def train(self, states, q_targets, batch_size=64):
        self.model.fit(
            states,
            q_targets,
            batch_size=batch_size,
            epochs=1,
            verbose=0
        )

    def copy_weights_from(self, other):
        for v_self, v_other in zip(
            self.model.trainable_variables,
            other.model.trainable_variables
        ):
            v_self.assign(v_other.numpy())

    # FIXED SAVE
    def save(self, path):
        if not path.endswith(".weights.h5"):
            path += ".weights.h5"

        self.model.save_weights(path)

    # FIXED LOAD
    def load(self, path):
        if not path.endswith(".weights.h5"):
            path += ".weights.h5"

        self.model.load_weights(path)


# ─────────────────────────────────────────────────────────────
# DDQN Agent
# ─────────────────────────────────────────────────────────────

class DDQNAgent:

    def __init__(
        self,
        alpha,
        gamma,
        n_actions,
        epsilon,
        batch_size,
        input_dims,
        epsilon_dec=0.9995,
        epsilon_end=0.10,
        mem_size=25000,
        replace_target=50,
        fname="ddqn_weights"
    ):

        self.action_space = list(
            range(n_actions)
        )

        self.n_actions = n_actions
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end

        self.batch_size = batch_size
        self.replace_target = replace_target

        self.fname = fname

        self.learn_step = 0

        self.memory = ReplayBuffer(
            mem_size,
            input_dims,
            n_actions
        )

        self.brain_eval = Brain(
            input_dims,
            n_actions,
            lr=alpha
        )

        self.brain_target = Brain(
            input_dims,
            n_actions,
            lr=alpha
        )

        # Initial sync
        self.brain_target.copy_weights_from(
            self.brain_eval
        )


    # ---------------------------------------------------------
    # Store transition
    # ---------------------------------------------------------

    def remember(
        self,
        state,
        action,
        reward,
        new_state,
        done
    ):
        self.memory.store(
            state,
            action,
            reward,
            new_state,
            done
        )


    # ---------------------------------------------------------
    # Epsilon-greedy action
    # ---------------------------------------------------------

    def choose_action(self, state):

        if np.random.random() < self.epsilon:
            return np.random.choice(
                self.action_space
            )

        state = np.array(
            state,
            dtype=np.float32
        )[np.newaxis, :]

        q_vals = self.brain_eval.predict(
            state
        )

        return int(
            np.argmax(q_vals[0])
        )


    # ---------------------------------------------------------
    # Learning
    # ---------------------------------------------------------

    def learn(self):

        if self.memory.mem_cntr < self.batch_size:
            return

        (
            states,
            actions,
            rewards,
            next_states,
            terminals
        ) = self.memory.sample(
            self.batch_size
        )

        action_indices = np.argmax(
            actions,
            axis=1
        )

        # Eval net chooses best action
        q_eval_next = self.brain_eval.predict(
            next_states
        )

        best_actions = np.argmax(
            q_eval_next,
            axis=1
        )

        # Target net evaluates
        q_target_next = self.brain_target.predict(
            next_states
        )

        q_pred = self.brain_eval.predict(
            states
        )

        batch_idx = np.arange(
            self.batch_size,
            dtype=np.int32
        )

        # DDQN target
        q_pred[
            batch_idx,
            action_indices
        ] = (
            rewards
            + self.gamma
            * q_target_next[
                batch_idx,
                best_actions
            ]
            * terminals
        )

        self.brain_eval.train(
            states,
            q_pred,
            batch_size=self.batch_size
        )

        # Decay epsilon
        self.epsilon = max(
            self.epsilon * self.epsilon_dec,
            self.epsilon_min
        )

        self.learn_step += 1

        if (
            self.learn_step
            % self.replace_target
            == 0
        ):
            self.update_network_parameters()


    # ---------------------------------------------------------
    # Sync target network
    # ---------------------------------------------------------

    def update_network_parameters(self):
        self.brain_target.copy_weights_from(
            self.brain_eval
        )


    # ---------------------------------------------------------
    # Save / Load
    # ---------------------------------------------------------

    def save_model(self):
        self.brain_eval.save(
            f"{self.fname}_eval"
        )

        self.brain_target.save(
            f"{self.fname}_target"
        )

        print(
            "[DDQN] Model saved."
        )


    def load_model(self):

        try:
            self.brain_eval.load(
                f"{self.fname}_eval"
            )

            self.brain_target.load(
                f"{self.fname}_target"
            )

            print(
                "[DDQN] Model loaded."
            )

        except Exception as e:
            print(
                f"[DDQN] Could not load model: {e}"
            )