import numpy as np
from abc import ABC, abstractmethod

class BaseFA(ABC):
    """
    Abstract Base Class for Function Approximation in RPI/CRPI.
    """
    
    @abstractmethod
    def evaluate_policy(self, P_mu: np.ndarray, r: np.ndarray, gamma: float, f_k: np.ndarray) -> np.ndarray:
        """
        Solves the constrained optimization problem to find the new value estimate f_{k+1}.
        
        Args:
            P_mu (np.ndarray): The SA x SA transition matrix for the current policy mu.
            r (np.ndarray): The SA x 1 reward vector.
            gamma (float): The discount factor.
            f_k (np.ndarray): The SA x 1 vector of the previous value estimate.
            
        Returns:
            np.ndarray: The SA x 1 vector representing the new value estimate f_{k+1}.
        """
        pass