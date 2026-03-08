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

        # max_r = np.max(np.abs(r))
        # epsilon = max(1e-4, max_r * 0.05)

        constraints = [
            f >= f_k - epsilon,
            (np.eye(self.SA) - gamma * P_mu) @ f <= r + epsilon
        ]

        objective = cp.Maximize(cp.sum(f))
        prob = cp.Problem(objective, constraints)
        
        # try:
        #     prob.solve(solver=cp.SCS, max_iters=2000)
        # except Exception:
        #     return f_k.reshape(-1, 1)

        # if prob.status not in ["optimal", "optimal_inaccurate"] or theta_scaled.value is None:
        #     return f_k.reshape(-1, 1)

        # try:
        #     # Force the highly accurate ECOS solver
        #     prob.solve(solver=cp.ECOS)
            
        #     # THE FIX: We absolutely CANNOT accept "optimal_inaccurate"
        #     if prob.status != "optimal":
        #         raise ValueError(f"Optimization failed. Status: {prob.status}")
                
        # except Exception as e:
        #     # If it fails, fall back to f_k to guarantee the lower bound is maintained
        #     return f_k.reshape(-1, 1)

        solvers_to_try = []


        if 'CLARABEL' in cp.installed_solvers():
            solvers_to_try.append(('CLARABEL', {}))

        # SCS: first-order ADMM, robust to near-degenerate problems, good fallback
        if 'SCS' in cp.installed_solvers():
            solvers_to_try.append(('SCS', {'max_iters': 10000, 'eps': 1e-5}))

        # GLPK: simplex-based LP solver; works on the boundary so thin feasible
        # sets near convergence are handled naturally
        if 'GLPK' in cp.installed_solvers():
            solvers_to_try.append(('GLPK', {}))

        # ECOS last: original solver, kept as last resort
        if 'ECOS' in cp.installed_solvers():
            solvers_to_try.append(('ECOS', {}))

        for solver_name, solver_kwargs in solvers_to_try:
            try:
                prob.solve(solver=getattr(cp, solver_name), **solver_kwargs)
                if prob.status in ('optimal', 'optimal_inaccurate') and theta_scaled.value is not None:
                    # Recover: f = h * scale + f_k = (Phi_n @ theta_h.value) * scale + f_k
                    f_val = Phi_scaled @ theta_scaled.value
                    return f_val.reshape(-1, 1), True
            except Exception as e:
                continue

            
        # All solvers failed — return f_k to preserve the monotonicity guarantee.
        # This should be extremely rare given the cascade above.
        return f_k.reshape(-1, 1), False

        f_val = Phi_scaled @ theta_scaled.value
        return f_val.reshape(-1, 1)