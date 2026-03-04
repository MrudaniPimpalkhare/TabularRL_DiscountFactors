import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from .linear_fa import LinearFA

class PolynomialFA(LinearFA):
    def __init__(self, Phi_raw: np.ndarray, degree: int):
        """
        Polynomial Function Approximation.
        
        Args:
            Phi_raw (np.ndarray): The raw state-action feature matrix of shape (SA, d_raw).
            degree (int): The maximum degree of the polynomial features.
        """
        self.degree = degree
        
        # Generate polynomial and interaction features.
        # include_bias=True automatically adds a column of 1s (intercept).
        poly = PolynomialFeatures(degree=self.degree, include_bias=True)
        Phi_poly = poly.fit_transform(Phi_raw)
        
        # Initialize the parent LinearFA class with the newly expanded feature matrix
        super().__init__(Phi=Phi_poly)