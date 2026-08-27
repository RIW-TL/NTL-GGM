"""TransMTGLasso Solver Module.

This module implements the Transfer Multi-Task Graphical Lasso (TransMTGLasso) algorithm,
which estimates task-specific precision matrices with a shared component via ADMM.
"""

import numpy as np
import time

class TransMTGLasso:
    """The TransMTGLasso solver class.

    This class implements the Transfer Multi-Task Graphical Lasso (TransMTGLasso) algorithm
    using the Alternating Direction Method of Multipliers (ADMM) for optimization.

    Attributes:
        covs (np.ndarray): A (K+1) × d × d array of empirical covariance matrices.
        penal_param_lasso (float): Lasso regularization parameter.
        alphas (np.ndarray): A (K+1)-dimensional array of task-specific weights.
        penal_param_admm (float): ADMM penalty parameter.
        eps_abs (float): Absolute tolerance for convergence.
        eps_rel (float): Relative tolerance for convergence.
        max_iter (int): Maximum number of ADMM iterations.

        n_iter (int): Number of iterations completed.
        residual_primal (list of float): Sequence of primal residuals.
        residual_dual (list of float): Sequence of dual residuals.
        eps_primal (list of float): Sequence of primal residual thresholds.
        eps_dual (list of float): Sequence of dual residual thresholds.

        precision_matrices (np.ndarray): A (K+1) × d × d array of estimated precision matrices.
        opt_shared (np.ndarray): A d × d matrix representing the estimated shared component.
        opt_diffs (np.ndarray): A (K+1) × d × d array of task-specific deviations.
        opt_duals (np.ndarray): A (K+1) × d × d array of dual variables used in ADMM.

        loss (list of float): Sequence of objective function values during optimization.
        time_used (float): Total runtime of the training procedure, in seconds.
    """

    def __init__(self, covs, penal_param_lasso, alphas, penal_param_admm, eps_abs=1e-4, eps_rel=1e-4, max_iter=5000):
        """Initialize the TransMTGLasso solver.

        Args:
            covs (np.ndarray): A (K+1) × d × d array of empirical covariance matrices.
            penal_param_lasso (float): Lasso penalty parameter.
            alphas (np.ndarray): A (K+1)-dimensional array of task-specific weights.
            penal_param_admm (float): ADMM penalty parameter.
            eps_abs (float, optional): Absolute tolerance for convergence. Default is 1e-4.
            eps_rel (float, optional): Relative tolerance for convergence. Default is 1e-4.
            max_iter (int, optional): Maximum number of ADMM iterations. Default is 5000.
        """
        # Save user-defined inputs
        self.covs = covs
        self.penal_param_lasso = penal_param_lasso
        self.penal_param_admm = penal_param_admm
        self.alphas = alphas
        self.eps_abs = eps_abs
        self.eps_rel = eps_rel
        self.max_iter = max_iter

        # Initialize optimization bookkeeping
        self.n_iter = None
        self.residual_primal = None
        self.residual_dual = None
        self.eps_primal = None
        self.eps_dual = None

        # Model parameters and internal variables
        self.precision_matrices = None      # Estimated precision matrices (K+1) × d × d
        self.opt_shared = None              # Shared component (d × d)
        self.opt_diffs = None               # Task-specific differential components (K+1) × d × d
        self.opt_duals = None               # Dual variables (K+1) × d × d

        # Training diagnostics
        self.loss = None
        self.time_used = None

    @staticmethod
    def soft_threshold(x, thresh):
        """Apply soft-thresholding to a scalar input.

        This function implements the proximal operator for the L1 norm:
            prox_{thresh * |·|}(x) = sign(x) * max(|x| - thresh, 0)

        Args:
            x (float): Input scalar.
            thresh (float): Non-negative threshold value.

        Returns:
            float: Soft-thresholded value.
        """
        if x > thresh:
            return x - thresh
        elif x < -thresh:
            return x + thresh
        else:
            return 0.0

    def sub_prob_solver(self, c, x_init=None, y_init=None, max_iter_sub=10000, print_info=False):
        """Solve the subproblem for updating opt_shared and opt_diffs.

        This routine solves the following convex optimization problem:
            minimize_{x, y}  (ρ/2) * ||x + y - c||_2^2 + λ|x| + λ ∑_k √α_k |y_k|
        using block coordinate descent with soft-thresholding updates.

        Args:
            c (np.ndarray): A (K+1,)-dimensional array representing the input vector.
            x_init (float, optional): Initial value of x. Defaults to 0.0 if None.
            y_init (np.ndarray, optional): Initial value of y. Defaults to zeros if None.
            max_iter_sub (int, optional): Maximum number of iterations. Default is 10000.
            print_info (bool, optional): If True, print optimization diagnostics.

        Returns:
            x (float): Estimated scalar shared component.
            y (np.ndarray): Estimated (K+1,)-dimensional task-specific deviations.
        """
        start_time_sub = time.time()

        # Number of tasks
        K = len(c) - 1

        # Initialize variables
        x = x_init if x_init is not None else 0.0
        y = y_init.copy() if y_init is not None else np.zeros(K + 1)

        # Initialize residual and tolerance
        residual_sub = float("inf")
        eps_sub = 0.0

        eps_abs_sub = self.eps_abs
        eps_rel_sub = self.eps_rel

        n_iter_sub = 0
        while residual_sub > eps_sub:
            n_iter_sub += 1

            if n_iter_sub > max_iter_sub:
                print(f"Warning! Subproblem did not converge after {max_iter_sub} iterations. "
                      f"Residual = {residual_sub:.4f}, Eps = {eps_sub:.4f}")
                return x, y

            # Update x using the average of (c - y), followed by soft-thresholding
            x = self.soft_threshold((c - y).mean(),
                                    self.penal_param_lasso / (self.penal_param_admm * (K + 1)))

            # Store current y for computing dual residual
            y_prev = y.copy()

            # Update each y_k using soft-thresholding
            for k in range(K + 1):
                y[k] = self.soft_threshold(
                    c[k] - x,
                    self.penal_param_lasso * np.sqrt(self.alphas[k]) / self.penal_param_admm
                )

            # Compute residual and stopping threshold
            residual_sub = self.penal_param_admm * abs((y - y_prev).sum())
            eps_sub = (K + 1) * eps_abs_sub + eps_rel_sub * self.penal_param_admm * max(
                abs(y).sum(), abs(y_prev).sum()
            )

            # Optional logging
            if print_info:
                loss_sub = (self.penal_param_admm / 2) * np.sum((x + y - c) ** 2)
                loss_sub += self.penal_param_lasso * abs(x)
                loss_sub += self.penal_param_lasso * np.sum(np.sqrt(self.alphas) * abs(y))
                print(f"Sub Problem | Iter: {n_iter_sub:4d} | Loss: {loss_sub:.5f} "
                      f"| Residual: {residual_sub:.5f} | Eps: {eps_sub:.5f} "
                      f"| Time: {time.time() - start_time_sub:.3f}s")

        return x, y

    def train(self, init_opt_shared=None, init_opt_diffs=None, init_opt_duals=None,
              vary_admm_penal=False, print_info=True, max_penal_param_admm=2.0):
        """Run the TransMTGLasso training algorithm using ADMM.

        Args:
            init_opt_shared (np.ndarray, optional): Initial d × d shared precision matrix.
                Defaults to identity matrix if None.
            init_opt_diffs (np.ndarray, optional): Initial (K+1) × d × d task-specific deviations.
                Defaults to all zeros if None.
            init_opt_duals (np.ndarray, optional): Initial (K+1) × d × d dual variables.
                Defaults to identity matrices if None.
            vary_admm_penal (bool, optional): Whether to use adaptive ADMM penalty tuning.
                The paper uses a fixed penalty, which is the stable default here.
            print_info (bool, optional): Whether to print progress at each iteration.
            max_penal_param_admm (float, optional): Maximum allowed ADMM penalty.
        """
        # Initialize tracking variables
        start_time = time.time()
        self.n_iter = 0
        self.residual_primal = [float("inf")]
        self.residual_dual = [float("inf")]
        self.eps_primal = [0.0]
        self.eps_dual = [0.0]
        self.loss = []

        K = self.covs.shape[0] - 1
        d = self.covs.shape[1]

        # Initialize primal variables
        self.opt_shared = init_opt_shared if init_opt_shared is not None else np.eye(d)
        self.opt_diffs = init_opt_diffs if init_opt_diffs is not None else np.zeros((K+1, d, d))

        # Initialize dual variables
        if init_opt_duals is not None:
            self.opt_duals = init_opt_duals
        else:
            self.opt_duals = np.array([np.eye(d) for _ in range(K+1)])

        self.precision_matrices = np.zeros((K+1, d, d))

        # Setup adaptive ADMM penalty strategy
        if vary_admm_penal:
            mu_penal_admm = 10.0
            tau_increase_admm = tau_decrease_admm = 2.0

        if print_info:
            print(f"penal_param_lasso: {self.penal_param_lasso}")

        C_check = np.zeros((K+1, d, d))

        # ADMM optimization loop
        while self.residual_primal[-1] > self.eps_primal[-1] or self.residual_dual[-1] > self.eps_dual[-1]:
            self.n_iter += 1

            if self.n_iter > self.max_iter:
                print("Warning! ADMM did not converge after {} iterations.".format(self.n_iter - 1))
                print("Final primal residual: {:.5f}, eps: {:.5f}".format(self.residual_primal[-1], self.eps_primal[-1]))
                print("Final dual residual: {:.5f}, eps: {:.5f}".format(self.residual_dual[-1], self.eps_dual[-1]))
                break

            opt_shared_prev = self.opt_shared.copy()
            opt_diffs_prev = self.opt_diffs.copy()

            # Update each task's precision matrix
            for k in range(K+1):
                C = -self.opt_duals[k] + self.opt_shared + self.opt_diffs[k] \
                    - (self.alphas[k] / self.penal_param_admm) * self.covs[k]

                eigvals, eigvecs = np.linalg.eigh(self.penal_param_admm * C)
                new_eigvals = (eigvals + np.sqrt(eigvals**2 + 4 * self.penal_param_admm * self.alphas[k])) \
                              / (2 * self.penal_param_admm)
                self.precision_matrices[k] = eigvecs @ np.diag(new_eigvals) @ eigvecs.T

                C_check[k] = self.precision_matrices[k] + self.opt_duals[k]

            # Update shared and task-specific terms via subproblem
            for i in range(d):
                for j in range(d):
                    self.opt_shared[i, j], self.opt_diffs[:, i, j] = self.sub_prob_solver(
                        c=C_check[:, i, j],
                        x_init=self.opt_shared[i, j],
                        y_init=self.opt_diffs[:, i, j].copy()
                    )

            # Symmetrize shared and task-specific estimates
            self.opt_shared = 0.5 * (self.opt_shared + self.opt_shared.T)
            for k in range(K+1):
                self.opt_diffs[k] = 0.5 * (self.opt_diffs[k] + self.opt_diffs[k].T)

            # Dual variable update
            self.opt_duals += self.penal_param_admm * (
                self.precision_matrices - (self.opt_shared + self.opt_diffs)
            )

            # Compute primal/dual residuals and thresholds
            self.residual_primal.append(np.linalg.norm(
                self.precision_matrices - (self.opt_shared + self.opt_diffs)))
            self.residual_dual.append(self.penal_param_admm * np.linalg.norm(
                (self.opt_shared + self.opt_diffs) - (opt_shared_prev + opt_diffs_prev)))

            norm_pmat = np.linalg.norm(self.precision_matrices)
            norm_sum = np.linalg.norm(self.opt_shared + self.opt_diffs)
            self.eps_primal.append(self.eps_abs * d * np.sqrt(K+1) + self.eps_rel * max(norm_pmat, norm_sum))
            self.eps_dual.append(self.eps_abs * d * np.sqrt(K+1) + self.eps_rel * np.linalg.norm(self.opt_duals))

            # Print iteration diagnostics
            if print_info:
                print(f"iteration round: {self.n_iter} | primal residual: {self.residual_primal[-1]:.3f} | "
                      f"dual residual: {self.residual_dual[-1]:.3f} | "
                      f"primal eps: {self.eps_primal[-1]:.3f} | dual eps: {self.eps_dual[-1]:.3f} | "
                      f"penalty param of ADMM: {self.penal_param_admm:.3f} | "
                      f"time used: {time.time() - start_time:.3f}(s)")

            # Adjust ADMM penalty if enabled
            if vary_admm_penal:
                if self.residual_primal[-1] > mu_penal_admm * self.residual_dual[-1]:
                    self.penal_param_admm *= tau_increase_admm
                elif self.residual_dual[-1] > mu_penal_admm * self.residual_primal[-1]:
                    self.penal_param_admm /= tau_decrease_admm

            self.penal_param_admm = min(self.penal_param_admm, max_penal_param_admm)

        # Remove initial placeholder values
        self.residual_primal = self.residual_primal[1:]
        self.residual_dual = self.residual_dual[1:]
        self.eps_primal = self.eps_primal[1:]
        self.eps_dual = self.eps_dual[1:]

        # Final symmetrization
        for k in range(K+1):
            self.opt_diffs[k] = 0.5 * (self.opt_diffs[k] + self.opt_diffs[k].T)
            self.precision_matrices[k] = 0.5 * (self.precision_matrices[k] + self.precision_matrices[k].T)
            self.opt_duals[k] = 0.5 * (self.opt_duals[k] + self.opt_duals[k].T)
        self.opt_shared = 0.5 * (self.opt_shared + self.opt_shared.T)

        # Record total runtime
        self.time_used = time.time() - start_time
