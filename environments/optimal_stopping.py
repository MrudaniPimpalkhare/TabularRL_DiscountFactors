import numpy as np
import gymnasium as gym
from gymnasium import spaces

class OptimalStoppingEnv(gym.Env):
    """
    Optimal Stopping Environment conforming to the Gymnasium API.
    Designed for use with tabular and FA-based Policy Iteration methods.
    """
    def __init__(self, N=100, cost=0.1, p_up=0.4, p_down=0.6):
        super().__init__()
        
        self.N = N 
        self.terminal_state = N
        self.cost = cost
        
        self.observation_space = spaces.Discrete(self.N + 1)
        self.action_space = spaces.Discrete(2)
        
        self.n_states = self.N + 1
        self.n_actions = 2
        
        self.P = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.R = np.zeros((self.n_states, self.n_actions))

        self.R[:, 1] = 0.0 
        
        # 2. Create a sparse "spike" only at the very top of the chain
        self.R[self.N - 1, 1] = 100.0 
        self.R[self.N - 2, 1] = 50.0 
        
        # 3. Small continuation cost to encourage moving, but not so high that it forces immediate stopping
        self.R[:, 0] = -self.cost
        
        for s in range(self.N):
            self.P[s, 1, self.terminal_state] = 1.0
            # self.R[s, 1] = s
            
            # self.R[s, 0] = -self.cost
            
            if s == 0:
                self.P[s, 0, 0] = 1.0 - p_up
                self.P[s, 0, 1] = p_up
            elif s == self.N - 1:
                self.P[s, 0, self.N - 1] = 1.0 - p_down
                self.P[s, 0, self.N - 2] = p_down
            else:
                self.P[s, 0, s - 1] = p_down
                self.P[s, 0, s + 1] = p_up
                self.P[s, 0, s] = 1.0 - p_up - p_down
                
        self.P[self.terminal_state, :, self.terminal_state] = 1.0
        self.R[self.terminal_state, :] = 0.0

        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.N // 2
        return self.state, {}

    def step(self, action):
        probs = self.P[self.state, action]
        next_state = self.np_random.choice(self.n_states, p=probs)
        reward = self.R[self.state, action]
        
        self.state = next_state
        terminated = (self.state == self.terminal_state)
        truncated = False
        
        return self.state, reward, terminated, truncated, {}

    def get_mdp_matrices(self):
        return self.P, self.R