"""
D-Trace Loss Solver

This module implements the solver for the D-Trace loss estimator, a method used for 
estimating differential networks between two populations. The optimization is performed 
via a proximal gradient descent algorithm, with cross-validation and BIC-based procedures 
for tuning parameter selection.

References:
    Yuan et al. (2017) - "Differential network analysis via lasso penalized D-trace loss"
"""

import numpy as np
import time

class DTraceSolver:
    """
    DTraceSolver implements the proximal gradient descent algorithm for minimizing the D-Trace loss,
    a convex objective used in differential network estimation between two populations.

    This solver estimates the differential precision matrix (network) between a target and source
    distribution using penalized optimization, promoting sparsity via an ℓ₁ regularization term.

    Attributes:
        cov_target (np.ndarray): A (d x d) empirical covariance matrix from the target data.
        cov_source (np.ndarray): A (d x d) empirical covariance matrix from the source data.
        penal_param (float): The regularization parameter for the ℓ₁ penalty.
        learning_rate (float): The step size used in the proximal gradient updates.
        eps_abs (float): Absolute tolerance for stopping criterion.
        eps_rel (float): Relative tolerance for stopping criterion.
        max_iter (int): Maximum number of allowed iterations.

        n_iter (int): Total number of iterations performed.
        residual (List[float]): Sequence of residual norms across iterations.
        eps (List[float]): Sequence of stopping thresholds used per iteration.
        diff_network (np.ndarray): The final estimated differential precision matrix (d x d).
        loss (List[float]): Sequence of D-Trace loss values across iterations.
        time_used (float): Total training time in seconds.
    """
    
    def __init__(self, cov_target, cov_source, penal_param,
                 learning_rate=None, eps_abs=1e-4, eps_rel=1e-4, max_iter=50):
        """
        Initialize a DTraceSolver instance.

        This class solves the D-Trace loss minimization problem using proximal gradient descent.
        It estimates the sparse differential precision matrix between the target and source distributions.

        Args:
            cov_target (np.ndarray): A (d x d) empirical covariance matrix from the target data.
            cov_source (np.ndarray): A (d x d) empirical covariance matrix from the source data.
            penal_param (float): ℓ₁ regularization parameter for promoting sparsity in the differential network.
            learning_rate (float, optional): Step size for proximal gradient updates.
                If None, it is set to 0.9 divided by the product of the operator norms of cov_target and cov_source.
            eps_abs (float): Absolute tolerance for stopping criterion. Default is 1e-4.
            eps_rel (float): Relative tolerance for stopping criterion. Default is 1e-4.
            max_iter (int): Maximum number of proximal gradient iterations. Default is 10,000.
        """
        self.cov_target = cov_target
        self.cov_source = cov_source
        self.penal_param = penal_param
        self.learning_rate = learning_rate
        self.eps_abs = eps_abs
        self.eps_rel = eps_rel
        self.max_iter = max_iter

        # Internal state variables for optimization
        self.n_iter = None                # Total iterations performed
        self.residual = None             # Sequence of primal residuals
        self.eps = None                  # Sequence of stopping thresholds
        self.diff_network = None         # Estimated differential network (d x d)
        self.loss = None                 # Objective values across iterations
        self.time_used = None            # Total computation time (seconds)

    def train(self, init_diff_network=None, print_info=True):
        """
        Run the proximal gradient descent algorithm to estimate the differential network.

        Args:
            init_diff_network (np.ndarray, optional): A (d x d) matrix used as the initial estimate.
                If None, initializes with the zero matrix. Default is None.
            print_info (bool): Whether to print training progress and diagnostics. Default is True.
        """
        start_time = time.time()
        self.n_iter = 0
        self.residual = [float("inf")]
        self.eps = [0.0]
        self.loss = []

        d = self.cov_source.shape[0]

        # Initialize differential network
        self.diff_network = init_diff_network if init_diff_network is not None else np.zeros((d, d))

        # Set learning rate if not provided
        if self.learning_rate is None:
            norm_target = np.linalg.norm(self.cov_target, ord=2)
            norm_source = np.linalg.norm(self.cov_source, ord=2)
            self.learning_rate = 0.9 / (norm_target * norm_source)

        if print_info:
            print(f"penal_param: {self.penal_param:.5f}")

        # Main optimization loop
        while self.residual[-1] > self.eps[-1]:
            self.n_iter += 1

            if self.n_iter > self.max_iter:
                print(f"Warning: Did not converge within {self.max_iter} iterations.")
                print(f"Final residual: {self.residual[-1]:.3e} | threshold: {self.eps[-1]:.3e}")
                break

            # Compute gradient step
            grad = 0.5 * (self.cov_target @ self.diff_network @ self.cov_source +
                          self.cov_source @ self.diff_network @ self.cov_target) - \
                   (self.cov_target - self.cov_source)
            A = self.diff_network - self.learning_rate * grad

            prev_diff_network = self.diff_network.copy()

            # Apply elementwise soft-thresholding (proximal operator)
            self.diff_network = np.sign(A) * np.maximum(np.abs(A) - self.penal_param * self.learning_rate, 0.0)

            # for i in range(d):
            #     for j in range(d):
            #         temp = abs(A[i, j]) - self.penal_param * self.learning_rate
            #         self.diff_network[i, j] = np.sign(A[i, j]) * temp if temp > 0 else 0.0

            change = self.diff_network - prev_diff_network

            # Compute residual and stopping criterion
            grad_change = (1 / self.learning_rate) * change - \
                          0.5 * self.cov_target @ change @ self.cov_source - \
                          0.5 * self.cov_source @ change @ self.cov_target
            curr_residual = np.linalg.norm(grad_change)
            self.residual.append(curr_residual)

            max_norm_term = max(
                (1 / self.learning_rate) * np.linalg.norm(change),
                0.5 * np.linalg.norm(self.cov_target @ change @ self.cov_source)
            )
            curr_eps = np.linalg.norm(self.eps_abs * d + self.eps_rel * max_norm_term)
            self.eps.append(curr_eps)

            # Compute current loss
            term1 = self.cov_target @ self.diff_network
            term2 = self.diff_network @ self.cov_source
            loss_val = 0.25 * np.trace(term1.T @ term2) + \
                       0.25 * np.trace((self.cov_source @ self.diff_network).T @ (self.diff_network @ self.cov_target)) - \
                       np.trace(self.diff_network.T @ (self.cov_target - self.cov_source))
            self.loss.append(loss_val)

            if print_info and self.n_iter % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Iteration {self.n_iter} | Loss: {loss_val:.4f} | Residual: {curr_residual:.4e} "
                      f"| Threshold: {curr_eps:.4e} | Time: {elapsed:.2f}s")

        # Finalize output
        self.residual = self.residual[1:]  # remove initialization
        self.eps = self.eps[1:]

        # Enforce symmetry
        self.diff_network = 0.5 * (self.diff_network + self.diff_network.T)

        self.time_used = time.time() - start_time
        if print_info:
            print(f"Total Time Used: {self.time_used:.2f} seconds")


class DTraceBIC():
    """
    DTraceBIC performs model selection for the D-Trace loss estimator using the Bayesian Information Criterion (BIC).

    This class estimates a sparse differential network between a target and source population
    by minimizing the D-Trace loss with an ℓ₁ penalty. The optimal penalty parameter is chosen 
    from a candidate list by minimizing the BIC score.

    Attributes:
        target_data (np.ndarray): A (n_target x d) matrix containing the target data samples.
        source_data (np.ndarray): A (n_source x d) matrix containing the source data samples.
        penal_param_list (List[float]): A list of candidate penalty parameters for ℓ₁ regularization.

        penal_param_bic (float): The selected penalty parameter based on the BIC criterion.
        time_used (float): Total computation time in seconds.
        bic_error (np.ndarray): A 1D array of BIC scores corresponding to each candidate parameter.
        best_model (np.ndarray): A (d x d) matrix representing the estimated differential network under the optimal BIC choice.

        learning_rate (float): Step size for proximal gradient updates.
        eps_abs (float): Absolute tolerance for stopping criterion.
        eps_rel (float): Relative tolerance for stopping criterion.
        max_iter (int): Maximum number of proximal gradient iterations.
    """

    def __init__(self, target_data, source_data, penal_param_list, 
                 learning_rate=None, eps_abs=1e-4, eps_rel=1e-4, max_iter=500):
        """
        Initialize the DTraceBIC instance for BIC-based model selection.

        This class estimates a sparse differential precision matrix between the 
        target and source distributions using the D-Trace loss. The optimal 
        regularization parameter is selected from a candidate list via the 
        Bayesian Information Criterion (BIC).

        Args:
            target_data (np.ndarray): A (n_target x d) array of samples from the target distribution.
            source_data (np.ndarray): A (n_source x d) array of samples from the source distribution.
            penal_param_list (List[float]): A list of candidate ℓ₁ penalty values to be evaluated by BIC.

            learning_rate (float, optional): Step size for the proximal gradient updates. If None,
                the algorithm will compute a default value based on the data.
            eps_abs (float): Absolute tolerance for convergence. Default is 1e-4.
            eps_rel (float): Relative tolerance for convergence. Default is 1e-4.
            max_iter (int): Maximum number of iterations for the optimization algorithm. Default is 10000.
        """
        self.target_data = target_data
        self.source_data = source_data
        self.penal_param_list = penal_param_list

        self.learning_rate = learning_rate
        self.eps_abs = eps_abs
        self.eps_rel = eps_rel
        self.max_iter = max_iter

        self.penal_param_bic = None  # Chosen penalty parameter via BIC
        self.time_used = None        # Time taken to run BIC selection
        self.bic_error = None        # Array of BIC scores for each penalty candidate
        self.best_model = None       # Differential network estimate under the best BIC penalty
    
    def bic(self, print_info=True):
        """
        Select the optimal lasso penalty parameter via BIC (Bayesian Information Criterion).

        Args:
            print_info (bool): Whether to print progress and timing information during training.
        """
        start_time = time.time()

        if print_info:
            print("BIC for DTrace Begins!")

        n_target, _ = self.target_data.shape
        n_source, _ = self.source_data.shape

        cov_target = np.cov(self.target_data.T)
        cov_source = np.cov(self.source_data.T)

        self.bic_error = np.full(len(self.penal_param_list), np.inf)

        prev_diff_network = None  # Used for warm-starting the solver
        min_bic_error = np.inf

        for j, pp in enumerate(self.penal_param_list):
            # Instantiate and train the solver
            dtrace = DTraceSolver(
                cov_target=cov_target,
                cov_source=cov_source,
                penal_param=pp,
                learning_rate=self.learning_rate,
                eps_abs=self.eps_abs,
                eps_rel=self.eps_rel,
                max_iter=self.max_iter
            )

            dtrace.train(init_diff_network=prev_diff_network, print_info=False)

            # Save result for next iteration warm-start
            prev_diff_network = dtrace.diff_network.copy()

            # Compute BIC score
            err = (
                0.5 * (cov_source @ dtrace.diff_network @ cov_target +
                       cov_target @ dtrace.diff_network @ cov_source)
                - (cov_target - cov_source)
            )
            est_sparse_level = (np.abs(dtrace.diff_network) > 0.0).sum()
            penalty = est_sparse_level * np.log(n_target + n_source) / (n_target + n_source)
            self.bic_error[j] = np.linalg.norm(err) + penalty

            # Track best model
            if self.bic_error[j] < min_bic_error:
                min_bic_error = self.bic_error[j]
                self.best_model = dtrace.diff_network

            if print_info:
                elapsed = time.time() - start_time
                print(f"Penalty Parameter: {pp:.5f} | BIC Error: {self.bic_error[j]:.3f} | Time Used: {elapsed:.3f}(s)")

        self.penal_param_bic = self.penal_param_list[np.argmin(self.bic_error)]

        if print_info:
            print(f"Final chosen penalty parameter: {self.penal_param_bic}")

        self.time_used = time.time() - start_time


class DTraceCV():
    """
    DTraceCV performs model selection for the D-Trace loss estimator using cross-validation.

    This class estimates a sparse differential precision matrix between two populations by 
    minimizing the D-Trace loss with an ℓ₁ penalty. The optimal penalty parameter is selected 
    from a candidate list via K-fold cross-validation.

    Attributes:
        target_data (np.ndarray): A (n_target x d) matrix containing samples from the target distribution.
        source_data (np.ndarray): A (n_source x d) matrix containing samples from the source distribution.
        n_fold (int): Number of folds used for cross-validation.
        penal_param_list (List[float]): List of candidate ℓ₁ penalty values to evaluate.

        penal_param_cv (float): The penalty parameter selected via cross-validation.
        time_used (float): Total computation time for the cross-validation procedure.
        cv_error (np.ndarray): A (n_fold x len(penal_param_list)) matrix storing the CV error for each fold and penalty.

        learning_rate (float): Step size for proximal gradient updates.
        eps_abs (float): Absolute tolerance for convergence.
        eps_rel (float): Relative tolerance for convergence.
        max_iter (int): Maximum number of proximal gradient iterations allowed.
    """

    def __init__(self, target_data, source_data, n_fold, penal_param_list,
                 learning_rate=None, eps_abs=1e-4, eps_rel=1e-4, max_iter=10000):
        """
        Initialize the DTraceCV instance for cross-validation-based model selection.

        This class estimates a sparse differential precision matrix using the D-Trace loss
        and selects the optimal ℓ₁ penalty parameter from a candidate list via K-fold cross-validation.

        Args:
            target_data (np.ndarray): A (n_target x d) matrix of target samples.
            source_data (np.ndarray): A (n_source x d) matrix of source samples.
            n_fold (int): Number of folds for cross-validation.
            penal_param_list (List[float]): List of candidate ℓ₁ penalty values.

            learning_rate (float, optional): Step size for the proximal gradient method.
                If None, it will be computed automatically based on data.
            eps_abs (float): Absolute tolerance for convergence. Default is 1e-4.
            eps_rel (float): Relative tolerance for convergence. Default is 1e-4.
            max_iter (int): Maximum number of optimization iterations. Default is 10,000.
        """
        self.target_data = target_data
        self.source_data = source_data
        self.n_fold = n_fold
        self.penal_param_list = penal_param_list

        self.learning_rate = learning_rate
        self.eps_abs = eps_abs
        self.eps_rel = eps_rel
        self.max_iter = max_iter

        self.penal_param_cv = None     # Final penalty selected via cross-validation
        self.time_used = None          # Total time used for cross-validation
        self.cv_error = None           # Cross-validation error matrix (n_fold x len(penal_param_list))

    def cv(self, print_info=True):
        """
        Select the optimal lasso penalty parameter via K-fold cross-validation.

        Args:
            print_info (bool): Whether to print progress and timing information during training.
        """
        start_time = time.time()

        if print_info:
            print("CV for DTrace Begins!")

        n_target, _ = self.target_data.shape
        n_source, _ = self.source_data.shape

        cut_points_target = np.linspace(0, n_target, num=self.n_fold + 1, dtype=int)
        cut_points_source = np.linspace(0, n_source, num=self.n_fold + 1, dtype=int)

        self.cv_error = np.full((self.n_fold, len(self.penal_param_list)), np.inf)

        for k in range(self.n_fold):
            # Target training/test split
            mask_target = np.ones(n_target, dtype=bool)
            mask_target[cut_points_target[k]:cut_points_target[k + 1]] = False
            target_train = self.target_data[mask_target, :]
            target_test = self.target_data[~mask_target, :]

            # Source training/test split
            mask_source = np.ones(n_source, dtype=bool)
            mask_source[cut_points_source[k]:cut_points_source[k + 1]] = False
            source_train = self.source_data[mask_source, :]
            source_test = self.source_data[~mask_source, :]

            prev_diff_network = None  # Warm start initialization

            for j, pp in enumerate(self.penal_param_list):
                # Compute training covariances
                cov_target_train = np.cov(target_train.T)
                cov_source_train = np.cov(source_train.T)

                # Train D-Trace solver
                dtrace = DTraceSolver(
                    cov_target=cov_target_train,
                    cov_source=cov_source_train,
                    penal_param=pp,
                    learning_rate=self.learning_rate,
                    eps_abs=self.eps_abs,
                    eps_rel=self.eps_rel,
                    max_iter=self.max_iter
                )
                dtrace.train(init_diff_network=prev_diff_network, print_info=False)

                # Update warm start
                prev_diff_network = dtrace.diff_network.copy()

                # Compute test covariances
                cov_target_test = np.cov(target_test.T)
                cov_source_test = np.cov(source_test.T)

                # Compute test error
                err = (
                    0.5 * (cov_source_test @ dtrace.diff_network @ cov_target_test +
                           cov_target_test @ dtrace.diff_network @ cov_source_test)
                    - (cov_target_test - cov_source_test)
                )
                self.cv_error[k, j] = np.linalg.norm(err)

                if print_info:
                    elapsed = time.time() - start_time
                    print(f"Fold: {k + 1} | Penalty Parameter: {pp:.5f} | CV Error: {self.cv_error[k, j]:.3f} | Time Used: {elapsed:.3f}(s)")

        # Select best penalty based on average error across folds
        self.penal_param_cv = self.penal_param_list[np.argmin(self.cv_error.mean(axis=1))]

        if print_info:
            print(f"Final chosen penalty parameter: {self.penal_param_cv}")

        self.time_used = time.time() - start_time
