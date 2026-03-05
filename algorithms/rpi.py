import numpy as np

class RPI:
    """
    Reliable Policy Iteration (RPI)[cite: 33].
    """
    def __init__(self, env, fa_model, gamma=0.9, max_iters=100):
        self.env = env
        self.fa_model = fa_model
        self.gamma = gamma
        self.max_iters = max_iters
        
        # Extract the exact transition and reward matrices
        self.P_env, self.R_env = env.get_mdp_matrices()
        self.S, self.A, _ = self.P_env.shape
        self.SA = self.S * self.A
        
        # Initialize uniform policy and value estimates
        self.mu = np.ones((self.S, self.A)) / self.A
        
        r_min = np.min(self.R_env)
        min_val = r_min / (1 - self.gamma)
        self.f_k = np.ones((self.SA, 1)) * min_val

        

        # Moved nu here so both RPI and CRPI can use it for metric tracking
        self.nu = np.ones((self.SA, 1)) / self.SA

    def get_P_mu(self, mu):
        """Constructs the (SA x SA) transition matrix for a given policy mu."""
        P_flat = self.P_env.reshape(self.SA, self.S)
        P_mu = np.zeros((self.SA, self.SA))
        for s_prime in range(self.S):
            for a_prime in range(self.A):
                P_mu[:, s_prime * self.A + a_prime] = P_flat[:, s_prime] * mu[s_prime, a_prime]
        return P_mu

    def train(self, track_metrics=True):
        r_flat = self.R_env.flatten().reshape(-1, 1)
        history = {'true_return': [], 'est_return': []}
        I = np.eye(self.SA)
        
        for k in range(self.max_iters):
            # 1. Build the SAxSA transition matrix for the current policy
            P_mu = self.get_P_mu(self.mu)
            
            # do initial check 
            # bellman_matrix = I - self.gamma * P_mu
            # if bellman_matrix @ self.f_k > r_flat:
            #     print(f"Warning: Initial f_k does not satisfy Bellman constraints at iteration {k+1}. Consider revising the initial f_k or checking the environment setup.")


            # 2. Policy Evaluation (Algorithm 1) [cite: 97, 98]
            self.f_k = self.fa_model.evaluate_policy(P_mu, r_flat, self.gamma, self.f_k)

            # Metric Tracking natively built-in
            if track_metrics:
                Q_true = np.linalg.inv(I - self.gamma * P_mu) @ r_flat
                history['true_return'].append((self.nu.T @ Q_true).item())
                history['est_return'].append((self.nu.T @ self.f_k).item())
            
            # 3. Policy Improvement (Greedy Update) [cite: 100, 103]
            Q = self.f_k.reshape(self.S, self.A)
            new_mu = np.zeros((self.S, self.A))
            best_actions = np.argmax(Q, axis=1)
            new_mu[np.arange(self.S), best_actions] = 1.0
            
            self.mu = new_mu
            
            print(f"RPI Iteration {k+1}/{self.max_iters} completed.")
            
        return self.mu, self.f_k, history