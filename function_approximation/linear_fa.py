import numpy as np
import cvxpy as cp
from .base_fa import BaseFA

class LinearFA(BaseFA):
    def __init__(self, n_states: int, n_actions: int, d_features: int):    
        self.d_features = d_features
        self.SA = n_states * n_actions
        Phi = np.random.uniform(1, 5, size=(self.SA, self.d_features))
        Phi[:, 0] = 1.0 # Bias column
        self.Phi = Phi
        # FIXED: self.d changed to self.d_features
        self.theta = cp.Variable(self.d_features) 

    def evaluate_policy(self, P_mu: np.ndarray, r: np.ndarray, gamma: float, f_k: np.ndarray) -> np.ndarray:
        norm_factor = np.linalg.norm(self.Phi, axis=0)
        norm_factor[norm_factor == 0] = 1.0 
        Phi_scaled = self.Phi / norm_factor

        # FIXED: self.d changed to self.d_features
        theta_scaled = cp.Variable(self.d_features)
        f = Phi_scaled @ theta_scaled 
        
        r = r.flatten()
        f_k = f_k.flatten()
        epsilon = 1e-4

        constraints = [
            f >= f_k - epsilon,
            (np.eye(self.SA) - gamma * P_mu) @ f <= r + epsilon
        ]

        objective = cp.Maximize(cp.sum(f))
        prob = cp.Problem(objective, constraints)
        
        try:
            prob.solve(solver=cp.SCS, max_iters=2000)
        except Exception:
            return f_k.reshape(-1, 1)

        if prob.status not in ["optimal", "optimal_inaccurate"] or theta_scaled.value is None:
            return f_k.reshape(-1, 1)

        f_val = Phi_scaled @ theta_scaled.value
        return f_val.reshape(-1, 1)