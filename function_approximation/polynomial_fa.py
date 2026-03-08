import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from .linear_fa import LinearFA
import cvxpy as cp

class PolynomialFA(LinearFA):
    def __init__(self, n_states: int, n_actions: int, degree: int, d_features: int = 3):
        self.degree = degree
        self.n_states = n_states
        self.n_actions = n_actions
        self.SA = n_states * n_actions
        self.d_features = d_features
        

        n_extra = d_features - 3

        Phi_backbone = np.zeros((self.SA, 3))
        row_idx = 0
        for s in range(self.n_states):
            for a in range(self.n_actions):
                Phi_backbone[row_idx, 0] = 1.0  # bias
                Phi_backbone[row_idx, 1] = s     # state index
                Phi_backbone[row_idx, 2] = a     # action index
                row_idx += 1
                
        # FIXED: Create exactly 3 columns to prevent polynomial noise explosion
        # self.Phi_raw = np.random.uniform(1, 5, size=(self.SA, self.d_features))
        # row_idx = 0
        # for s in range(self.n_states):
        #     for a in range(self.n_actions):
        #         self.Phi_raw[row_idx, 0] = 1.0  
        #         self.Phi_raw[row_idx, 1] = s    
        #         self.Phi_raw[row_idx, 2] = a    
        #         row_idx += 1

        Phi_random = np.random.uniform(1, 5, size=(self.SA, n_extra))
        poly = PolynomialFeatures(degree=self.degree, include_bias=True)
        Phi_poly = poly.fit_transform(Phi_backbone)

        self.Phi = np.hstack([Phi_poly, Phi_random])
        
        # poly = PolynomialFeatures(degree=self.degree, include_bias=True)
        # self.Phi = poly.fit_transform(self.Phi_raw)
        
        # FIXED: Update dimension so evaluate_policy creates the right sized Variable
        self.d_features = self.Phi.shape[1] 
        self.theta = cp.Variable(self.d_features)