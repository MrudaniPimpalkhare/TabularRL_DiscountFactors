import numpy as np
import gymnasium as gym
from gymnasium import spaces

class ChainWalkEnv(gym.Env):
    """
    Chain Walk Environment conforming to the Gymnasium API.
    """
    def __init__(self, N=50, p=0.9, time_limit=500):
        super().__init__()
        
        self.N = N
        self.p = p
        self.time_limit = time_limit

        # States are 0 to N-1
        self.observation_space = spaces.Discrete(self.N)
        self.action_space = spaces.Discrete(2) # 0: Left, 1: Right
        
        self.n_states = self.N
        self.n_actions = 2
        
        # Target states located N/4 from each end
        self.target_1 = int(self.N / 4)
        self.target_2 = self.N - 1 - int(self.N / 4)
        
        self.P = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.R = np.zeros((self.n_states, self.n_actions))
        
        self._build_mdp()
        self.state = None

        self.counter = 0

    def _build_mdp(self):
        for s in range(self.N):
            for a in [0, 1]:
                if a == 0: # Intend Left
                    intended_s = max(0, s - 1)
                    opposite_s = min(self.N - 1, s + 1)
                else:      # Intend Right
                    intended_s = min(self.N - 1, s + 1)
                    opposite_s = max(0, s - 1)
                    
                self.P[s, a, intended_s] += self.p
                self.P[s, a, opposite_s] += (1.0 - self.p)
                
        # Reward is +1 upon entering the target states
        for s in range(self.N):
            for a in [0, 1]:
                prob_enter_t1 = self.P[s, a, self.target_1]
                prob_enter_t2 = self.P[s, a, self.target_2]
                self.R[s, a] = (prob_enter_t1 + prob_enter_t2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Uniform initial distribution over the states
        self.state = self.np_random.integers(0, self.N)
        self.counter = 0
        return self.state, {}

    def step(self, action):
        self.counter += 1
        probs = self.P[self.state, action]
        next_state = self.np_random.choice(self.n_states, p=probs)

        # Get reward for the current state-action pair
        reward = self.R[self.state, action]
        
        self.state = next_state
        terminated = False # The chain can be traversed indefinitely
        truncated = (self.counter >= self.time_limit) # Truncate after a certain number of steps to prevent infinite episodes
        
        return self.state, reward, terminated, truncated, {}

    def get_mdp_matrices(self):
        return self.P, self.R