import numpy as np
import matplotlib.pyplot as plt

def solve_tabular_pi(env, gamma=0.9, max_iters=100):
    """Computes the exact optimal policy and Q-values using standard Tabular PI."""
    P, R = env.get_mdp_matrices()
    S, A, _ = P.shape
    SA = S * A
    
    # Initialize arbitrary policy
    mu = np.ones((S, A)) / A
    r_flat = R.flatten().reshape(-1, 1)
    
    for _ in range(max_iters):
        # Build P_mu
        P_flat = P.reshape(SA, S)
        P_mu = np.zeros((SA, SA))
        for s_prime in range(S):
            for a_prime in range(A):
                P_mu[:, s_prime * A + a_prime] = P_flat[:, s_prime] * mu[s_prime, a_prime]
                
        # Exact Policy Evaluation: Q_mu = (I - gamma * P_mu)^-1 * r
        I = np.eye(SA)
        Q_mu = np.linalg.inv(I - gamma * P_mu) @ r_flat
        
        # Greedy Improvement
        Q_reshaped = Q_mu.reshape(S, A)
        new_mu = np.zeros((S, A))
        best_actions = np.argmax(Q_reshaped, axis=1)
        new_mu[np.arange(S), best_actions] = 1.0
        
        if np.array_equal(mu, new_mu):
            break
        mu = new_mu
        
    return mu, Q_mu