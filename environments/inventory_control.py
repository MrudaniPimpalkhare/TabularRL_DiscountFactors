import numpy as np
import gymnasium as gym
from gymnasium import spaces

class InventoryControlEnv(gym.Env):
    """
    Inventory Control Environment exactly as described in Section 5.1
    of 'Monotone and Conservative Policy Iteration Beyond the Tabular Case'.
    """
    def __init__(self, M=49, p=10.0, c=5.0, h=1.0, gamma=0.9):
        super().__init__()
        self.M = M          # Capacity (M=49 -> 50 states)
        self.p = p          # Selling price
        self.c = c          # Purchase cost
        self.h = h          # Holding cost
        self.gamma = gamma

        self.n_states = M + 1
        self.n_actions = M + 1

        self.observation_space = spaces.Discrete(self.n_states)
        self.action_space = spaces.Discrete(self.n_actions)

        self.P, self.R = self._build_mdp()

    def _build_mdp(self):
        """Builds P(S, A, S) and R(S, A) using Uniform demand and Action Projection."""
        P = np.zeros((self.n_states, self.n_actions, self.n_states))
        R = np.zeros((self.n_states, self.n_actions))

        # Demand is Uniform over {0, ..., M}
        prob_d = 1.0 / (self.M + 1)

        for s in range(self.n_states):
            for a_raw in range(self.n_actions):
                # Action Projection: Ensure s + a <= M
                # This prevents the 'penalty cliff' that ruins FA training.
                a = min(a_raw, self.M - s)
                y = s + a  # Inventory after ordering

                expected_reward = 0
                for d in range(self.M + 1):
                    # Dynamics: s_next = max(0, y - d)
                    s_next = max(0, y - d)
                    P[s, a_raw, s_next] += prob_d

                    # Reward: p * min(d, y) - c * a - h * max(0, y - d)
                    revenue = self.p * min(d, y)
                    cost = self.c * a
                    holding = self.h * max(0, y - d)

                    expected_reward += prob_d * (revenue - cost - holding)

                R[s, a_raw] = expected_reward
        return P, R

    def get_mdp_matrices(self):
        return self.P, self.R
