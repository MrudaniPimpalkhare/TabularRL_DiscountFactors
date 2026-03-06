# filepath: /home/sans/TabularRL_DiscountFactors/utils.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from algorithms.rpi import RPI
from algorithms.crpi import CRPI


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


def run_experiment(env, fa_class, gamma=0.9, iters=50, n_seeds=10, 
                   fa_params=None, title_suffix="", verbose=True):
    """
    Run RPI and CRPI experiments with the given function approximation class.
    
    Args:
        env: The environment instance (must have get_mdp_matrices method)
        fa_class: The function approximation class (LinearFA or PolynomialFA)
        gamma: Discount factor
        iters: Number of iterations per training run
        n_seeds: Number of random seeds to run
        fa_params: Dict of parameters to pass to fa_class constructor 
                   (e.g., {'n_states': 10, 'n_actions': 2, 'd_features': 4})
        title_suffix: String to append to plot titles (e.g., "Linear FA" or "Poly Degree 3")
        verbose: Whether to print progress messages
        
    Returns:
        dict: Results containing metrics and history
    """
    if fa_params is None:
        fa_params = {}
    
    P, R = env.get_mdp_matrices()
    n_states = P.shape[0]
    n_actions = P.shape[1]
    SA = n_states * n_actions
    nu = np.ones((SA, 1)) / SA
    
    # Get optimal baseline
    _, Q_opt = solve_tabular_pi(env, gamma)
    optimal_return = (nu.T @ Q_opt).item()
    
    # Storage for all runs
    rpi_all_true, crpi_all_true = [], []
    rpi_seed0_true, rpi_seed0_est = None, None
    crpi_seed0_true, crpi_seed0_est = None, None
    
    if verbose:
        print(f"Running {n_seeds} seeds for {title_suffix}...")
    
    for seed in range(n_seeds):
        np.random.seed(seed)
        
        # Create FA instances
        fa_rpi = fa_class(**fa_params)
        fa_crpi = fa_class(**fa_params)
        
        # Train RPI
        rpi_agent = RPI(env, fa_rpi, gamma, iters)
        _, _, r_hist = rpi_agent.train(track_metrics=True)
        rpi_all_true.append(r_hist['true_return'])
        
        # Train CRPI
        crpi_agent = CRPI(env, fa_crpi, gamma, iters)
        _, _, c_hist = crpi_agent.train(track_metrics=True)
        crpi_all_true.append(c_hist['true_return'])
        
        # Save seed 0 for detailed plot
        if seed == 0:
            rpi_seed0_true = r_hist['true_return']
            rpi_seed0_est = r_hist['est_return']
            crpi_seed0_true = c_hist['true_return']
            crpi_seed0_est = c_hist['est_return']
    
    rpi_all_true = np.array(rpi_all_true)
    crpi_all_true = np.array(crpi_all_true)
    
    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.suptitle(f"RPI vs CRPI: {title_suffix}", fontsize=14, fontweight='bold')
    
    # Subplot 1: Single Run (Seed 0)
    ax1.plot(rpi_seed0_true, color='#2ca02c', label='RPI True Return', linewidth=2)
    ax1.plot(rpi_seed0_est, color='#2ca02c', linestyle='--', label='RPI Lower Bound')
    ax1.plot(crpi_seed0_true, color='#000000', label='CRPI True Return', linewidth=2)
    ax1.plot(crpi_seed0_est, color='#000000', linestyle='--', label='CRPI Lower Bound')
    ax1.axhline(optimal_return, color='gray', linewidth=2, linestyle=':', label='Optimal')
    ax1.set_title("Single Run (Seed 0)")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Return")
    ax1.legend(fontsize='small')
    ax1.grid(alpha=0.3)
    
    # Subplot 2: Aggregated Performance
    rpi_mean = np.mean(rpi_all_true, axis=0)
    rpi_std = np.std(rpi_all_true, axis=0)
    crpi_mean = np.mean(crpi_all_true, axis=0)
    crpi_std = np.std(crpi_all_true, axis=0)
    
    ax2.plot(rpi_mean, color='#2ca02c', label='RPI Mean', linewidth=2)
    ax2.fill_between(range(iters), rpi_mean - rpi_std, rpi_mean + rpi_std, 
                     color='#2ca02c', alpha=0.2)
    ax2.plot(crpi_mean, color='#000000', label='CRPI Mean', linewidth=2)
    ax2.fill_between(range(iters), crpi_mean - crpi_std, crpi_mean + crpi_std, 
                     color='#000000', alpha=0.2)
    ax2.axhline(optimal_return, color='gray', linewidth=2, linestyle=':', label='Optimal')
    ax2.set_title(f"Aggregated Performance ({n_seeds} Seeds)")
    ax2.set_xlabel("Iterations")
    ax2.legend(fontsize='small')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # --- Metrics Table ---
    rpi_aucs = [np.trapezoid(run) for run in rpi_all_true]
    crpi_aucs = [np.trapezoid(run) for run in crpi_all_true]
    
    metrics_data = {
        "Algorithm": ["RPI", "CRPI"],
        "AUC (mean ± std)": [
            f"{np.mean(rpi_aucs):.1f} ± {np.std(rpi_aucs):.1f}",
            f"{np.mean(crpi_aucs):.1f} ± {np.std(crpi_aucs):.1f}"
        ],
        "Terminal Perf. (mean ± std)": [
            f"{rpi_mean[-1]:.2f} ± {rpi_std[-1]:.2f}",
            f"{crpi_mean[-1]:.2f} ± {crpi_std[-1]:.2f}"
        ]
    }
    df = pd.DataFrame(metrics_data)
    display(df)
    
    return {
        'rpi_all_true': rpi_all_true,
        'crpi_all_true': crpi_all_true,
        'rpi_mean': rpi_mean,
        'crpi_mean': crpi_mean,
        'optimal_return': optimal_return,
        'metrics_df': df
    }


def run_polynomial_sweep(env, poly_fa_class, degrees, gamma=0.9, iters=50, 
                         n_seeds=5, base_fa_params=None, verbose=True):
    """
    Run experiments across multiple polynomial degrees.
    
    Args:
        env: The environment instance
        poly_fa_class: The PolynomialFA class
        degrees: List of polynomial degrees to test
        gamma: Discount factor
        iters: Number of iterations per training run
        n_seeds: Number of random seeds per degree
        base_fa_params: Dict of base params for PolynomialFA (n_states, n_actions, d_features)
        verbose: Whether to print progress
        
    Returns:
        dict: Summary results across all degrees
    """
    if base_fa_params is None:
        base_fa_params = {}
    
    P, R = env.get_mdp_matrices()
    n_states = P.shape[0]
    n_actions = P.shape[1]
    SA = n_states * n_actions
    nu = np.ones((SA, 1)) / SA
    
    # Get optimal baseline
    _, Q_opt = solve_tabular_pi(env, gamma)
    optimal_return = (nu.T @ Q_opt).item()
    
    all_results = {}
    summary_data = {'Degree': [], 'RPI Terminal': [], 'CRPI Terminal': []}
    
    for deg in degrees:
        if verbose:
            print(f"\n{'='*50}")
            print(f"POLYNOMIAL DEGREE: {deg}")
            print('='*50)
        
        rpi_all_true, crpi_all_true = [], []
        rpi_seed0_true, rpi_seed0_est = None, None
        crpi_seed0_true, crpi_seed0_est = None, None
        
        for seed in range(n_seeds):
            np.random.seed(seed)
            
            # Create FA with current degree
            fa_params = {**base_fa_params, 'degree': deg}
            fa_rpi = poly_fa_class(**fa_params)
            fa_crpi = poly_fa_class(**fa_params)
            
            # Train RPI
            rpi_agent = RPI(env, fa_rpi, gamma, iters)
            _, _, r_hist = rpi_agent.train(track_metrics=True)
            rpi_all_true.append(r_hist['true_return'])
            
            # Train CRPI
            crpi_agent = CRPI(env, fa_crpi, gamma, iters)
            _, _, c_hist = crpi_agent.train(track_metrics=True)
            crpi_all_true.append(c_hist['true_return'])
            
            if seed == 0:
                rpi_seed0_true = r_hist['true_return']
                rpi_seed0_est = r_hist['est_return']
                crpi_seed0_true = c_hist['true_return']
                crpi_seed0_est = c_hist['est_return']
        
        rpi_all_true = np.array(rpi_all_true)
        crpi_all_true = np.array(crpi_all_true)
        
        # --- Plotting for this degree ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
        fig.suptitle(f"Polynomial Degree {deg}", fontsize=14, fontweight='bold')
        
        # Subplot 1: Single Run
        ax1.plot(rpi_seed0_true, color='#2ca02c', label='RPI True Return', linewidth=2)
        ax1.plot(rpi_seed0_est, color='#2ca02c', linestyle='--', label='RPI Lower Bound')
        ax1.plot(crpi_seed0_true, color='#000000', label='CRPI True Return', linewidth=2)
        ax1.plot(crpi_seed0_est, color='#000000', linestyle='--', label='CRPI Lower Bound')
        ax1.axhline(optimal_return, color='gray', linewidth=2, linestyle=':', label='Optimal')
        ax1.set_title("Single Run (Seed 0)")
        ax1.set_xlabel("Iterations")
        ax1.set_ylabel("Return")
        ax1.legend(fontsize='small')
        ax1.grid(alpha=0.3)
        
        # Subplot 2: Aggregated
        rpi_mean = np.mean(rpi_all_true, axis=0)
        rpi_std = np.std(rpi_all_true, axis=0)
        crpi_mean = np.mean(crpi_all_true, axis=0)
        crpi_std = np.std(crpi_all_true, axis=0)
        
        ax2.plot(rpi_mean, color='#2ca02c', label='RPI Mean', linewidth=2)
        ax2.fill_between(range(iters), rpi_mean - rpi_std, rpi_mean + rpi_std,
                         color='#2ca02c', alpha=0.15)
        ax2.plot(crpi_mean, color='#000000', label='CRPI Mean', linewidth=2)
        ax2.fill_between(range(iters), crpi_mean - crpi_std, crpi_mean + crpi_std,
                         color='#000000', alpha=0.15)
        ax2.axhline(optimal_return, color='gray', linewidth=2, linestyle=':', label='Optimal')
        ax2.set_title(f"Aggregated ({n_seeds} Seeds)")
        ax2.set_xlabel("Iterations")
        ax2.legend(fontsize='small')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Metrics table for this degree
        rpi_aucs = [np.trapezoid(run) for run in rpi_all_true]
        crpi_aucs = [np.trapezoid(run) for run in crpi_all_true]
        
        metrics_data = {
            "Algorithm": ["RPI", "CRPI"],
            "AUC (mean ± std)": [
                f"{np.mean(rpi_aucs):.1f} ± {np.std(rpi_aucs):.1f}",
                f"{np.mean(crpi_aucs):.1f} ± {np.std(crpi_aucs):.1f}"
            ],
            "Terminal Perf. (mean ± std)": [
                f"{rpi_mean[-1]:.2f} ± {rpi_std[-1]:.2f}",
                f"{crpi_mean[-1]:.2f} ± {crpi_std[-1]:.2f}"
            ]
        }
        display(pd.DataFrame(metrics_data))
        
        # Store results
        all_results[deg] = {
            'rpi_all_true': rpi_all_true,
            'crpi_all_true': crpi_all_true,
            'rpi_mean': rpi_mean,
            'crpi_mean': crpi_mean
        }
        
        summary_data['Degree'].append(deg)
        summary_data['RPI Terminal'].append(f"{rpi_mean[-1]:.3f}")
        summary_data['CRPI Terminal'].append(f"{crpi_mean[-1]:.3f}")
    
    # Final summary table
    print("\n" + "="*50)
    print("SUMMARY ACROSS ALL DEGREES")
    print("="*50)
    summary_df = pd.DataFrame(summary_data)
    display(summary_df)
    
    return all_results, summary_df
