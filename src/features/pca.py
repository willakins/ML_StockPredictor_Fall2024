"""
PCA algorithm class
Do not copy for your homework 3
"""
import numpy as np

class PCA(object):

    def __init__(self):
        self.U = None
        self.S = None
        self.V = None

    def fit(self, X: np.ndarray) ->None:
        """		
        Decompose dataset into principal components by finding the singular value decomposition of the centered dataset X

        Args: 
            X: (N,D) numpy array corresponding to a dataset
        
        Return:
            None
        
        Set:
            self.U: (N, min(N,D)) numpy array
            self.S: (min(N,D), ) numpy array
            self.V: (min(N,D), D) numpy array
        """
        X_bar = self.center(X, 0)
        U, S, V_t = np.linalg.svd(X_bar, full_matrices = False)
        
        self.U = U
        self.S = S
        self.V = V_t
        

    def transform(self, data: np.ndarray, K: int) ->np.ndarray:
        """		
        Transform data to reduce the number of features such that final data (X_new) has K features (columns)
        
        Args:
            data: (N,D) numpy array corresponding to a dataset
            K: int value for number of columns to be kept
        
        Return:
            X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data
        """
        X_bar = self.center(data, a = 0)
        V = self.V.T
        principle_directions = V[:, :K]
        X_transformed = np.dot(X_bar, principle_directions)
        return X_transformed
    
    def center(data: np.ndarray, a: int):
        return data - np.mean(data, axis = a)