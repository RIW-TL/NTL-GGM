"""
TransGLasso Solver Module

This module implements the core solver for the Trans-Glasso algorithm, a two-step 
transfer learning method for precision matrix estimation. It integrates multi-task 
graphical lasso (Trans-MT-Glasso) and differential network estimation (D-Trace loss).

References:
- Authors et al. (2025), "Trans-Glasso: A Transfer Learning Approach to Precision Matrix Estimation"
"""

import time
import numpy as np

try:
    # Package import (the layout used by the original project).
    from .dtrace import *
    from .transmtglasso import *
except ImportError:
    # Standalone import, used when this directory is called from R/reticulate.
    from dtrace import *
    from transmtglasso import *


class TransGLasso:
    """
    TransGLasso Solver Class

    This class implements the full Trans-Glasso procedure for precision matrix estimation 
    via transfer learning. It integrates two main components: differential network estimation 
    (D-Trace loss) and multi-task graphical lasso (Trans-MT-Glasso). 

    Attributes:
        covs (np.ndarray): A (K+1) × d × d array of empirical covariance matrices, 
            where the first slice corresponds to the target domain and the rest to source domains.
        penal_param_dtrace (np.ndarray): A length-K array of Lasso penalty parameters 
            used in the D-Trace differential network estimators.
        penal_param_transmtglasso (float): Lasso penalty parameter for Trans-MT-Glasso.
        alphas (np.ndarray): A length-(K+1) array of normalized sample size weights for each domain.
        penal_param_admm (float): ADMM penalty parameter used to solve Trans-MT-Glasso.
        learning_rate_proxgd (float): Step size for the proximal gradient descent 
            used in solving the D-Trace optimization problem.
        eps_abs (float): Absolute tolerance for convergence.
        eps_rel (float): Relative tolerance for convergence.
        max_iter_dtrace (int): Maximum number of iterations for solving the D-Trace problem.
        max_iter_transmtglasso (int): Maximum number of iterations for solving Trans-MT-Glasso.

    Outputs (after calling `train()`):
        precision_matrices_mt (np.ndarray): A (K+1) × d × d array of estimated precision matrices 
            from Trans-MT-Glasso.
        precision_matrix_target (np.ndarray): A d × d matrix representing the final estimated 
            precision matrix for the target domain after refinement.
        diff_networks (np.ndarray): A K × d × d array of estimated differential networks 
            between the target and each source.
        time_used (float): Total runtime in seconds for the full estimation procedure.
    """

    def __init__(
        self,
        covs,
        penal_param_dtrace,
        penal_param_transmtglasso,
        alphas,
        penal_param_admm,
        learning_rate_proxgd=None,
        eps_abs=1e-4,
        eps_rel=1e-4,
        max_iter_dtrace=10000,
        max_iter_transmtglasso=10000,
    ):
        """
        Initialize a TransGLasso solver instance.

        Args:
            covs (np.ndarray): A (K+1) × d × d array of empirical covariance matrices. 
                The first matrix corresponds to the target domain; the rest are from source domains.
            penal_param_dtrace (np.ndarray): A length-K array of Lasso penalty parameters 
                used in the D-Trace estimators for differential networks.
            penal_param_transmtglasso (float): Lasso penalty parameter used in Trans-MT-Glasso.
            alphas (np.ndarray): A length-(K+1) array of normalized weights based on sample sizes.
            penal_param_admm (float): ADMM penalty parameter for solving Trans-MT-Glasso.
            learning_rate_proxgd (float, optional): Learning rate for proximal gradient descent 
                in the D-Trace loss minimization. Default is None.
            eps_abs (float): Absolute tolerance for convergence. Default is 1e-4.
            eps_rel (float): Relative tolerance for convergence. Default is 1e-4.
            max_iter_dtrace (int): Maximum iterations for solving D-Trace loss. Default is 10,000.
            max_iter_transmtglasso (int): Maximum iterations for solving Trans-MT-Glasso. Default is 10,000.
        """
        self.covs = covs
        self.penal_param_dtrace = penal_param_dtrace
        self.penal_param_transmtglasso = penal_param_transmtglasso
        self.alphas = alphas
        self.penal_param_admm = penal_param_admm
        self.learning_rate_proxgd = learning_rate_proxgd
        self.eps_abs = eps_abs
        self.eps_rel = eps_rel
        self.max_iter_dtrace = max_iter_dtrace
        self.max_iter_transmtglasso = max_iter_transmtglasso

        # Outputs initialized as None
        self.precision_matrices_mt = None
        self.diff_networks = None
        self.precision_matrix_target = None
        self.time_used = None

    def train(self):
        """Run the full TransGLasso estimation procedure."""
        start_time = time.time()

        K = self.covs.shape[0] - 1
        d = self.covs.shape[1]

        # Step 1: Estimate differential networks via D-Trace
        self.diff_networks = np.zeros((K, d, d))
        for k in range(K):
            print(f"Start estimating the differential network with respect to source {k + 1}")

            dtrace_instance = DTraceSolver(
                cov_target=self.covs[0],
                cov_source=self.covs[k + 1],
                penal_param=self.penal_param_dtrace[k],
                learning_rate=self.learning_rate_proxgd,
                eps_abs=self.eps_abs,
                eps_rel=self.eps_rel,
                max_iter=self.max_iter_dtrace
            )
            dtrace_instance.train()
            self.diff_networks[k] = dtrace_instance.diff_network.copy()
            print()

        # Step 2: Estimate shared precision matrices via Trans-MT-Glasso
        print("Start solving TransMTGLasso")

        transmtglasso = TransMTGLasso(
            covs=self.covs,
            penal_param_lasso=self.penal_param_transmtglasso,
            alphas=self.alphas,
            penal_param_admm=self.penal_param_admm,
            eps_abs=self.eps_abs,
            eps_rel=self.eps_rel,
            max_iter=self.max_iter_transmtglasso
        )
        transmtglasso.train()
        self.precision_matrices_mt = transmtglasso.precision_matrices

        print()

        # Step 3: Refine target precision matrix
        self.precision_matrix_target = self.alphas[0] * self.precision_matrices_mt[0]
        for k in range(K):
            self.precision_matrix_target += self.alphas[k + 1] * (
                self.precision_matrices_mt[k + 1] - self.diff_networks[k]
            )

        # Final: Record time
        end_time = time.time()
        self.time_used = end_time - start_time
        print(f"Total time used: {self.time_used:.3f}(s)")


class TransGLassoMS:
    """
    TransGLassoMS Solver Class with Model Selection

    This class implements the Trans-Glasso estimator with automatic model selection 
    for tuning parameters. It performs a two-stage procedure: first estimating differential 
    networks via D-Trace, and then estimating precision matrices via Trans-MT-Glasso.

    The Lasso penalty parameters for D-Trace are selected using BIC.
    The Lasso penalty parameter for Trans-MT-Glasso can be selected using either BIC or cross-validation.

    Attributes:
        target_data (np.ndarray): An (n_target × d) array representing the target dataset.
        source_data (list of np.ndarray): A list of K source datasets, where the k-th entry 
            is an (n_source_k × d) array.
        penal_param_dtrace_list (list of float): List of candidate Lasso penalties for D-Trace.
        penal_param_transmtglasso_list (list of float): List of candidate Lasso penalties for Trans-MT-Glasso.

        alphas (np.ndarray): A (K+1)-dimensional array of normalized sample size weights.
        penal_param_admm (float): ADMM penalty parameter used for solving Trans-MT-Glasso.
        learning_rate_proxgd (float): Learning rate for the proximal gradient method used in D-Trace.
        eps_abs (float): Absolute tolerance for convergence.
        eps_rel (float): Relative tolerance for convergence.
        max_iter_dtrace (int): Maximum number of iterations for solving the D-Trace loss.
        max_iter_transmtglasso (int): Maximum number of iterations for solving Trans-MT-Glasso.

        chosen_penal_param_dtrace (np.ndarray): A K-dimensional array containing the selected 
            Lasso penalties for D-Trace (via BIC).
        chosen_penal_param_transmtglasso (float): Selected Lasso penalty for Trans-MT-Glasso 
            (via BIC or CV).
        n_fold (int): Number of folds used in cross-validation (if applicable).

        precision_matrices_mt (np.ndarray): A (K+1 × d × d) array of estimated precision matrices 
            from Trans-MT-Glasso.
        precision_matrix_target (np.ndarray): A (d × d) matrix representing the final estimated 
            target precision matrix.
        diff_networks (np.ndarray): A (K × d × d) array of estimated differential networks.
        time_used (float): Total runtime (in seconds) for the model selection and estimation procedure.
        cv_error (np.ndarray): A (n_fold × len(penal_param_transmtglasso_list)) array where each entry 
            records the cross-validation error for a given fold and penalty.
        bic_error (np.ndarray): A length-l array recording the BIC values for each candidate 
            penalty parameter in Trans-MT-Glasso.
    """

    def __init__(
        self,
        target_data,
        source_data,
        diff_network_list,
        penal_param_dtrace_list,
        penal_param_transmtglasso_list,
        penal_param_admm=1.0,
        n_fold=5,
        learning_rate_proxgd=None,
        eps_abs=1e-4,
        eps_rel=1e-4,
        max_iter_dtrace=5000,
        max_iter_transmtglasso=5000,
    ):
        """
        Initialize a TransGLassoMS instance with model selection capabilities.

        Args:
            target_data (np.ndarray): A (n_target × d) array containing the target dataset.
            source_data (list of np.ndarray): A list of K source datasets, each with shape (n_source_k × d).
            penal_param_dtrace_list (list of float): Candidate Lasso penalties for D-Trace differential estimators.
            penal_param_transmtglasso_list (list of float): Candidate Lasso penalties for Trans-MT-Glasso.
            penal_param_admm (float): ADMM penalty parameter for Trans-MT-Glasso. Default is 1.0.
            n_fold (int): Number of cross-validation folds (for model selection). Default is 5.
            learning_rate_proxgd (float, optional): Step size for the proximal gradient method in D-Trace.
            eps_abs (float): Absolute tolerance for convergence. Default is 1e-4.
            eps_rel (float): Relative tolerance for convergence. Default is 1e-4.
            max_iter_dtrace (int): Maximum iterations for D-Trace optimization. Default is 5000.
            max_iter_transmtglasso (int): Maximum iterations for Trans-MT-Glasso optimization. Default is 5000.
        """
        self.target_data = target_data
        self.source_data = source_data
        self.diff_network_list = diff_network_list
        self.penal_param_dtrace_list = penal_param_dtrace_list
        self.penal_param_transmtglasso_list = penal_param_transmtglasso_list

        self.penal_param_admm = penal_param_admm
        self.learning_rate_proxgd = learning_rate_proxgd
        self.eps_abs = eps_abs
        self.eps_rel = eps_rel
        self.max_iter_dtrace = max_iter_dtrace
        self.max_iter_transmtglasso = max_iter_transmtglasso

        self.n_fold = n_fold

        # Outputs to be populated during training
        self.diff_networks = None
        self.precision_matrix_target = None
        self.time_used = None
        self.chosen_penal_param_dtrace = None
        self.chosen_penal_param_transmtglasso = None
        self.cv_error = None
        self.bic_error = None

        # Compute normalized weights based on sample sizes
        n_target = self.target_data.shape[0]
        self.alphas = [n_target]
        for data_k in self.source_data:
            self.alphas.append(data_k.shape[0])
        self.alphas = np.array(self.alphas, dtype=float)
        self.alphas /= self.alphas.sum()

    def model_selction(self, method='BIC', print_info=True):
        """
        Perform model selection for Trans-GLasso.

        This method selects the Lasso penalty parameters used in D-Trace and Trans-MT-Glasso.
        - D-Trace penalties are always selected using BIC.
        - Trans-MT-Glasso penalties can be selected using either BIC (default) or cross-validation.

        Args:
            method (str): Model selection method for Trans-MT-Glasso. Must be either 'BIC' or 'CV'. Default is 'BIC'.
            print_info (bool): Whether to print training progress and diagnostic information. Default is True.
        """
        if method == 'CV':
            self.model_selction_cv(print_info=print_info)
        elif method == 'BIC':
            self.model_selction_bic(print_info=print_info)
        else:
            raise ValueError(f"Invalid method: {method}. Must be 'BIC' or 'CV'.")

    def model_selction_bic(self, print_info=True):
        """
        Perform model selection for Trans-GLasso using BIC.

        This method selects the optimal Lasso penalty parameters for:
        - D-Trace (via BIC)
        - Trans-MT-Glasso (via BIC)

        Args:
            print_info (bool): Whether to print training progress and timing info.
        """
        start_time = time.time()

        if print_info:
            print("Start model selection by BIC")

        n_target, d = self.target_data.shape
        K = len(self.source_data)
        n_total = n_target + sum(self.source_data[k].shape[0] for k in range(K))

        # -------------------- Step 1: Model selection for D-Trace (BIC) -------------------- #
        if print_info:
            print("Step 1: Model selection for D-Trace (BIC)")
        self.diff_networks = np.zeros((K, d, d))
        # self.chosen_penal_param_dtrace = np.zeros(K)

        for k in range(K):
              self.diff_networks[k] = self.diff_network_list[k]
        #     if print_info:
        #         print(f"Start model selection for DTrace with source {k + 1}")

        #     dtrace_bic = DTraceBIC(
        #         target_data=self.target_data,
        #         source_data=self.source_data[k],
        #         penal_param_list=self.penal_param_dtrace_list,
        #         eps_abs=self.eps_abs,
        #         eps_rel=self.eps_rel,
        #         max_iter=self.max_iter_dtrace
        #     )
        #     dtrace_bic.bic(print_info=print_info)
        #     self.chosen_penal_param_dtrace[k] = dtrace_bic.penal_param_bic
        #     self.diff_networks[k] = dtrace_bic.best_model

        # if print_info:
        #     time_now = time.time()
        #     print("Model selection for DTrace finished!")
        #     print(f"Total time used: {time_now - start_time:.3f}(s)\n")

        # -------------------- Step 2: Model selection for Trans-MT-Glasso (BIC) -------------------- #
        if print_info:
            print("Start model selection for Trans-MT-GLasso by BIC")

        self.bic_error = np.zeros(len(self.penal_param_transmtglasso_list))

        covs = np.zeros((K + 1, d, d))
        covs[0] = np.cov(self.target_data.T)
        for k in range(K):
            covs[k + 1] = np.cov(self.source_data[k].T)

        # Warm-start initialization
        prev_opt_shared = None
        prev_opt_diffs = None
        prev_opt_duals = None
        min_bic = float("inf")

        for j, pp in enumerate(self.penal_param_transmtglasso_list):
            start_time2 = time.time()

            transmtglasso = TransMTGLasso(
                covs=covs,
                penal_param_lasso=pp,
                alphas=self.alphas,
                penal_param_admm=self.penal_param_admm,
                eps_abs=self.eps_abs,
                eps_rel=self.eps_rel,
                max_iter=self.max_iter_transmtglasso
            )

            transmtglasso.train(
                init_opt_shared=prev_opt_shared,
                init_opt_diffs=prev_opt_diffs,
                init_opt_duals=prev_opt_duals,
                print_info=False
            )

            # Update warm-start variables
            prev_opt_shared = transmtglasso.opt_shared.copy()
            prev_opt_diffs = transmtglasso.opt_diffs.copy()
            prev_opt_duals = transmtglasso.opt_duals.copy()

            # Compute estimated target precision matrix
            precision_matrix_target = self.alphas[0] * transmtglasso.precision_matrices[0]
            for k in range(K):
                precision_matrix_target += self.alphas[k + 1] * (
                    transmtglasso.precision_matrices[k + 1] - self.diff_networks[k]
                )

            # Compute BIC
            prec_det = np.linalg.det(precision_matrix_target)
            sparsity = (np.abs(precision_matrix_target) > 0.0).sum()

            if prec_det > 0:
                self.bic_error[j] = (
                    np.trace(covs[0] @ precision_matrix_target)
                    - np.log(prec_det)
                    + sparsity * np.log(n_total) / n_total
                )
            else:
                self.bic_error[j] = sparsity * np.log(n_total) / n_total

            # Track best model
            if self.bic_error[j] < min_bic:
                self.precision_matrix_target = precision_matrix_target.copy()
                self.precision_matrices_mt = transmtglasso.precision_matrices.copy()
                self.chosen_penal_param_transmtglasso = pp
                min_bic = self.bic_error[j]

            if print_info:
                time_now = time.time()
                print(
                    f"Penalty Parameter: {pp:.5f} | BIC Error: {self.bic_error[j]:.3f} "
                    f"| Time Used: {time_now - start_time2:.3f}(s) | "
                    f"Total Time Used: {time_now - start_time:.3f}(s)"
                )

        if print_info:
            print(f"Final chosen penalty parameter for Trans-MT-GLasso: {self.chosen_penal_param_transmtglasso:.5f}")

        end_time = time.time()
        self.time_used = end_time - start_time

        if print_info:
            print("\n")
            print(f"Total time used: {self.time_used:.3f}(s)")
    
    def model_selction_cv(self, print_info=True):
        """Model selection for Trans-GLasso via cross-validation.

        Args:
            print_info (bool): Whether to print training and timing information.
        """
        start_time = time.time()

        if print_info:
            print("Start model selection")

        n_target, d = self.target_data.shape
        K = len(self.source_data)

        # -------------------- Step 1: Model selection for D-Trace (BIC) -------------------- #
        self.diff_networks = np.zeros((K, d, d))

        for k in range(K):
            self.diff_networks[k] = self.diff_network_list[k]

        # if print_info:
        #     print("Model selection for DTrace finished!")
        #     print(f"Total time used: {time.time() - start_time:.3f}(s)\n")

        # -------------------- Step 2: Model selection for Trans-MT-GLasso (CV) -------------------- #
        if print_info:
            print("Start model selection for Trans-MT-GLasso")

        self.cv_error = np.zeros((self.n_fold, len(self.penal_param_transmtglasso_list)))
        cut_points_target = np.linspace(0, n_target, num=self.n_fold + 1, dtype=int)

        covs_train = np.zeros((K + 1, d, d))
        for k in range(K):
            covs_train[k + 1] = np.cov(self.source_data[k].T)

        for k_fold in range(self.n_fold):
            # Prepare target training and test splits
            mask = np.ones(n_target, dtype=bool)
            mask[cut_points_target[k_fold]:cut_points_target[k_fold + 1]] = False
            target_train = self.target_data[mask]
            target_test = self.target_data[~mask]

            covs_train[0] = np.cov(target_train.T)

            # Warm-start initialization
            prev_opt_shared = None
            prev_opt_diffs = None
            prev_opt_duals = None

            for j, pp in enumerate(self.penal_param_transmtglasso_list):
                iter_start_time = time.time()

                transmtglasso = TransMTGLasso(
                    covs=covs_train,
                    penal_param_lasso=pp,
                    alphas=self.alphas,
                    penal_param_admm=self.penal_param_admm,
                    eps_abs=self.eps_abs,
                    eps_rel=self.eps_rel,
                    max_iter=self.max_iter_transmtglasso
                )

                transmtglasso.train(
                    init_opt_shared=prev_opt_shared,
                    init_opt_diffs=prev_opt_diffs,
                    init_opt_duals=prev_opt_duals,
                    print_info=False
                )

                # Update warm-start
                prev_opt_shared = transmtglasso.opt_shared.copy()
                prev_opt_diffs = transmtglasso.opt_diffs.copy()
                prev_opt_duals = transmtglasso.opt_duals.copy()

                # Compute target precision matrix
                precision_matrix_target = self.alphas[0] * transmtglasso.precision_matrices[0]
                for k in range(K):
                    precision_matrix_target += self.alphas[k + 1] * (
                        transmtglasso.precision_matrices[k + 1] - self.diff_networks[k]
                    )

                # Compute CV error
                cov_target_test = np.cov(target_test.T)
                prec_det = np.linalg.det(precision_matrix_target)
                if prec_det > 0:
                    self.cv_error[k_fold, j] = (
                        np.trace(cov_target_test @ precision_matrix_target) - np.log(prec_det)
                    )

                if print_info:
                    iter_end_time = time.time()
                    print(
                        f"Fold: {k_fold + 1} | Penalty Parameter: {pp:.5f} | "
                        f"CV Error: {self.cv_error[k_fold, j]:.3f} | "
                        f"Time Used: {iter_end_time - iter_start_time:.3f}(s) | "
                        f"Total Time: {iter_end_time - start_time:.3f}(s)"
                    )

        # Choose optimal penalty
        self.chosen_penal_param_transmtglasso = self.penal_param_transmtglasso_list[
            np.argmin(self.cv_error.mean(axis=1))
        ]

        if print_info:
            print(f"Final chosen penalty parameter for Trans-MT-GLasso: {self.chosen_penal_param_transmtglasso:.5f}")

        # -------------------- Step 3: Train final model -------------------- #
        if print_info:
            print("Start training the final model by solving Trans-MT-GLasso")

        covs_train[0] = np.cov(self.target_data.T)

        transmtglasso = TransMTGLasso(
            covs=covs_train,
            penal_param_lasso=self.chosen_penal_param_transmtglasso,
            alphas=self.alphas,
            penal_param_admm=self.penal_param_admm,
            eps_abs=self.eps_abs,
            eps_rel=self.eps_rel,
            max_iter=self.max_iter_transmtglasso
        )
        transmtglasso.train(print_info=False)
        self.precision_matrices_mt = transmtglasso.precision_matrices

        # Compute final target precision matrix
        self.precision_matrix_target = self.alphas[0] * self.precision_matrices_mt[0]
        for k in range(K):
            self.precision_matrix_target += self.alphas[k + 1] * (
                self.precision_matrices_mt[k + 1] - self.diff_networks[k]
            )

        self.time_used = time.time() - start_time

        if print_info:
            print("\n")
            print(f"Total time used: {self.time_used:.3f}(s)")
