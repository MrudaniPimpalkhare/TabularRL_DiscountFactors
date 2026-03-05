import numpy as np
from .rpi import RPI

class CRPI(RPI):
    """
    Conservative Reliable Policy Iteration (CRPI)[cite: 33].
    Inherits matrix builders and setups from RPI.
    """
    def __init__(self, env, fa_model, gamma=0.9, max_iters=100):
        super().__init__(env, fa_model, gamma, max_iters)
        # Uniform initial distribution nu over SA space 
        # self.nu = np.ones((self.SA, 1)) / self.SA - Moved to RPI for shared metric tracking

    def train(self, track_metrics=True):
        r_flat = self.R_env.flatten().reshape(-1, 1)
        history = {'true_return': [], 'est_return': []}
        I = np.eye(self.SA)
        
        for k in range(self.max_iters):
            P_mu = self.get_P_mu(self.mu)
            
            # 1. Policy Evaluation (Same as RPI) [cite: 132, 133]
            self.f_k = self.fa_model.evaluate_policy(P_mu, r_flat, self.gamma, self.f_k) # this is f_k+1 in the paper

            # Native Metric Tracking
            if track_metrics:
                Q_true = np.linalg.inv(I - self.gamma * P_mu) @ r_flat
                history['true_return'].append((self.nu.T @ Q_true).item())
                history['est_return'].append((self.nu.T @ self.f_k).item())
            
            # 2. Find the Greedy Policy (mu_bar) [cite: 138]
            Q = self.f_k.reshape(self.S, self.A)
            bar_mu = np.zeros((self.S, self.A)) # mu_bar
            best_actions = np.argmax(Q, axis=1)
            bar_mu[np.arange(self.S), best_actions] = 1.0
            
            P_bar_mu = self.get_P_mu(bar_mu)
            
            # --- Calculating step size alpha^* ---
            
            # a) Advantage Approximation: a_mu^bar_mu(f) = (P_bar_mu - P_mu) * f [cite: 163]
            a_mu_bar = (P_bar_mu - P_mu) @ self.f_k
            
            # b) Discounted state-action occupancy: d_mu^T = (1-gamma) * nu^T * (I - gamma * P_mu)^-1 
            # I = np.eye(self.SA)
            inv_matrix = np.linalg.inv(I - self.gamma * P_mu)
            d_mu_T = (1 - self.gamma) * self.nu.T @ inv_matrix
            
            # c) Expected Advantage: A_mu^bar_mu(f) = d_mu^T * a_mu_bar [cite: 163]
            A_mu_bar = (d_mu_T @ a_mu_bar).item()
            
            # d) Bellman operator applied to f: T_mu(f) = r + gamma * P_mu * f [cite: 84]
            T_mu_f = r_flat + self.gamma * P_mu @ self.f_k
            
            # e) FA Error term: a_mu_bar applied to (T_mu(f) - f) -> (P_bar_mu - P_mu)(T_mu(f) - f) [cite: 202]
            fa_error_vector = (P_bar_mu - P_mu) @ (T_mu_f - self.f_k)
            
            # f) Calculate eta_1 and eta_2 [cite: 202]
            eta_1 = (1 - self.gamma) * A_mu_bar
            eta_2 = ((1 - self.gamma)**2) * (self.nu.T @ fa_error_vector).item()
            
            # g) Calculate denominator partial
            # State occupancy delta_mu = d_mu^T * P 
            P_state = self.P_env.reshape(self.SA, self.S)
            delta_mu = (d_mu_T @ P_state).flatten() 
            
            # Total Variation distance: ||bar_mu - mu||_{1, delta_mu} 
            tv_dist = np.sum(delta_mu * np.sum(np.abs(bar_mu - self.mu), axis=1))
            
            # Span semi-norm 
            span_a = np.max(a_mu_bar) - np.min(a_mu_bar)
            
            partial = self.gamma * tv_dist * span_a
            
            # h) Determine optimal mixture coefficients [cite: 197, 200, 203]
            if partial == 0:
                alpha_star = 1.0  # Fallback to greedy if no variation
            else:
                alpha_1_star = (eta_1 + eta_2) / partial
                alpha_0_star = eta_1 / partial
                
                if alpha_1_star > 0:
                    alpha_star = alpha_1_star
                else:
                    alpha_star = alpha_0_star
                    
            # 3. Policy Update (Conservative Mixture) [cite: 140, 141]
            alpha_k = min(1.0, max(0.0, alpha_star)) # clip between 0 and 1
            self.mu = alpha_k * bar_mu + (1 - alpha_k) * self.mu
            
            print(f"CRPI Iteration {k+1}/{self.max_iters} | alpha_k = {alpha_k:.4f}")
            
        return self.mu, self.f_k, history