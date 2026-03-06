import numpy as np
import cvxpy as cp
from .base_fa import BaseFA

class LinearFA(BaseFA):
    def __init__(self, Phi: np.ndarray):    
        """
        Linear Function Approximation.
        
        Args:
            Phi (np.ndarray): The feature matrix of shape (SA, d). SA = number of state-action pairs, d = number of features.
        """
        self.Phi = Phi
        self.SA, self.d = Phi.shape
        # theta is the weight vector we are trying to optimize
        self.theta = cp.Variable(self.d) # The solver will tweak these 5 variables (weights) to find the optimal solution.

    def evaluate_policy(self, P_mu: np.ndarray, r: np.ndarray, gamma: float, f_k: np.ndarray) -> np.ndarray:
        # 1. Scaling: Normalize Phi to prevent numerical blowup with high degrees
        # This keeps the math stable for the solver
        norm_factor = np.linalg.norm(self.Phi, axis=0)
        norm_factor[norm_factor == 0] = 1.0 # Avoid division by zero
        Phi_scaled = self.Phi / norm_factor

        # Define variable in scaled space
        theta_scaled = cp.Variable(self.d)
        f = Phi_scaled @ theta_scaled 
        
        r = r.flatten()
        f_k = f_k.flatten()
        epsilon = 1e-4

        # Constraints (using scaled features)
        constraints = [
            f >= f_k - epsilon,
            (np.eye(self.SA) - gamma * P_mu) @ f <= r + epsilon
        ]

        objective = cp.Maximize(cp.sum(f))
        prob = cp.Problem(objective, constraints)
        
        # 2. Multi-Solver Strategy
        # try:
        #     # Try ECOS first (Fast/Accurate)
        #     prob.solve(solver=cp.ECOS, max_iters=100)
        # except:
        try:
            # Fallback to SCS (More robust against high-degree polynomial noise)
            prob.solve(solver=cp.SCS, max_iters=2000)
        except Exception:
            return f_k.reshape(-1, 1)

        if prob.status not in ["optimal", "optimal_inaccurate"] or theta_scaled.value is None:
            return f_k.reshape(-1, 1)

        # Convert back from scaled space to return the true f values
        f_val = Phi_scaled @ theta_scaled.value
        return f_val.reshape(-1, 1)