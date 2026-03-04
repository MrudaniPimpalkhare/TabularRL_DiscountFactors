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
        """
        Solves the LP to find the weights theta that maximize the l1 distance 
        from f_k subject to the Bellman constraints.
        """
        # Define the function f = Phi * theta
        f = self.Phi @ self.theta 
        
        # Reshape inputs to ensure 1D arrays for CVXPY compatibility
        r = r.flatten()
        f_k = f_k.flatten()

        # Constraint 1: f >= f_k 
        # (This ensures monotonic reliability under FA)
        constraint_1 = f >= f_k

        # Constraint 2: T_mu(f) >= f  =>  (I - gamma * P_mu) @ f <= r
        I = np.eye(self.SA)
        bellman_matrix = I - gamma * P_mu
        constraint_2 = bellman_matrix @ f <= r

        constraints = [constraint_1, constraint_2]

        # Objective: Maximize the l1-norm of (f - f_k). 
        # Since constraint_1 guarantees f >= f_k, all terms are positive.
        # Thus, maximizing sum(f - f_k) is equivalent to maximizing sum(f).
        objective = cp.Maximize(cp.sum(f))

        # Define and solve the problem
        prob = cp.Problem(objective, constraints)
        
        try:
            # ECOS and SCS are good open-source solvers included with cvxpy
            prob.solve(solver=cp.ECOS)
            
            # this should not be the case according to claim 3.1 
            if prob.status not in ["optimal", "optimal_inaccurate"]:
                raise ValueError(f"Optimization failed. Status: {prob.status}")
                
        except Exception as e:
            print(f"Warning: LP Solver failed with error: {e}. Returning f_k as fallback.")
            return f_k.reshape(-1, 1) # Fallback to prevent complete crash

        # Compute the final f_{k+1} using the optimized theta
        f_k_plus_1 = self.Phi @ self.theta.value
        return f_k_plus_1.reshape(-1, 1)