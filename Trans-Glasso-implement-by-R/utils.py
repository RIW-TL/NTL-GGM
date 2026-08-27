"""Utility classes and functions for Trans-Glasso."""

import numpy as np

class Divergence:
    """
    Class to compute divergence distances as defined in Li et al. (2023).

    Attributes:
        target_prec (ndarray): A (d × d) numpy array representing the target precision matrix.
        source_precs (ndarray): A (K × d × d) numpy array representing the source precision matrices.
        divg_mats (ndarray): A (K × d × d) numpy array of divergence matrices.
        divg_distances (ndarray): A (K,) numpy array of divergence distances between the target
                                  and each source precision matrix.
        divg_dist (float): The maximum value among `divg_distances`, representing the overall divergence.
    """

    def __init__(self, target_prec, source_precs, ord):
        """
        Initialize the Divergence instance.

        Args:
            target_prec (ndarray): Target precision matrix of shape (d, d).
            source_precs (ndarray): Source precision matrices of shape (K, d, d).
            ord (float): Norm order used to compute divergence. Should be in [0, 1].
        """
        self.target_prec = target_prec
        self.source_precs = source_precs

        K = self.source_precs.shape[0]   # number of source tasks
        d = self.target_prec.shape[0]    # dimension of precision matrices

        # Initialize and compute divergence matrices
        self.divg_mats = np.zeros((K, d, d))
        for k in range(K):
            self.divg_mats[k] = self.target_prec @ np.linalg.inv(self.source_precs[k]) - np.eye(d)
            self.divg_mats[k] = np.round(self.divg_mats[k], decimals=3)

        # Compute divergence distances for each source
        self.divg_distances = np.zeros(K)
        for k in range(K):
            col_norm = np.max(np.linalg.norm(self.divg_mats[k], ord=ord, axis=0))
            row_norm = np.max(np.linalg.norm(self.divg_mats[k], ord=ord, axis=1))
            self.divg_distances[k] = col_norm + row_norm

        # Compute the overall divergence distance
        self.divg_dist = np.max(self.divg_distances)

def prediction_error(prec_mat, test_data, min_eigen=1e-3):
    """
    Compute the prediction error using normalized negative log-likelihood.

    Args:
        prec_mat (ndarray): A (d × d) precision matrix estimator.
        test_data (ndarray): A (n × d) matrix representing test data.
        min_eigen (float): Minimum eigenvalue threshold used for positive semi-definite projection.
    
    Returns:
        float: Normalized negative log-likelihood (prediction error).
    """
    d = prec_mat.shape[0]  # dimensionality

    # Symmetrize the precision matrix
    Omega = 0.5 * (prec_mat + prec_mat.T)

    # Project onto the PSD cone by thresholding eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(Omega)
    Omega = eigenvectors @ np.diag(np.maximum(eigenvalues, min_eigen)) @ eigenvectors.T

    # Compute empirical covariance matrix from test data
    emp_cov = np.cov(test_data.T)

    # Compute normalized negative log-likelihood
    nll = (np.trace(emp_cov @ Omega) - np.log(np.linalg.det(Omega))) / (2 * d)
    return nll + np.log(2 * np.pi) / 2
