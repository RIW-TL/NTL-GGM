# Trans-Glasso-CV for an unknown informative source set
#
# Main usage:
#   fit <- transglasso_unknownA(target_data, source_data)
#
# target_data is an n_target x p numeric matrix/data.frame. source_data is a
# list of numeric matrices/data.frames, each with the same p variables.

.transglasso_state <- new.env(parent = emptyenv())
.transglasso_state$modules <- NULL
.transglasso_state$key <- NULL

.transglasso_default_work_dir <- local({
  source_file <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
  if (is.null(source_file) || !nzchar(source_file)) {
    getwd()
  } else {
    dirname(source_file)
  }
})

.transglasso_find_python <- function(python = NULL) {
  if (!is.null(python)) {
    if (length(python) != 1L || !nzchar(python) || !file.exists(python)) {
      stop("`python` must point to an existing Python executable.", call. = FALSE)
    }
    return(normalizePath(python, winslash = "/", mustWork = TRUE))
  }

  configured <- Sys.getenv("RETICULATE_PYTHON", unset = "")
  if (nzchar(configured) && file.exists(configured)) {
    return(normalizePath(configured, winslash = "/", mustWork = TRUE))
  }

  if (.Platform$OS.type == "windows" && nzchar(Sys.which("py"))) {
    detected <- tryCatch(
      system2(
        Sys.which("py"),
        c("-3", "-c", shQuote("import sys; print(sys.executable)")),
        stdout = TRUE,
        stderr = FALSE
      ),
      error = function(e) character(0)
    )
    detected <- trimws(detected)
    detected <- detected[nzchar(detected) & file.exists(detected)]
    if (length(detected)) {
      return(normalizePath(detected[[1L]], winslash = "/", mustWork = TRUE))
    }
  }

  candidates <- unname(c(Sys.which("python3"), Sys.which("python")))
  candidates <- candidates[nzchar(candidates) & file.exists(candidates)]
  if (length(candidates)) {
    return(normalizePath(candidates[[1L]], winslash = "/", mustWork = TRUE))
  }

  stop(
    "No usable Python installation was found. Supply `python = '/path/to/python'`.",
    call. = FALSE
  )
}

.transglasso_python_packages_ok <- function(python) {
  output <- suppressWarnings(system2(
    python,
    c("-c", shQuote("import numpy, scipy, sklearn")),
    stdout = TRUE,
    stderr = TRUE
  ))
  status <- attr(output, "status")
  is.null(status) || identical(as.integer(status), 0L)
}

# Run once with install = TRUE on a new server. This installs reticulate and
# the Python packages numpy, scipy and scikit-learn if they are missing.
setup_transglasso_python <- function(
    python = NULL,
    install = FALSE,
    work_dir = .transglasso_default_work_dir) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    if (!isTRUE(install)) {
      stop(
        "R package `reticulate` is required. Run setup_transglasso_python(install = TRUE).",
        call. = FALSE
      )
    }
    install.packages("reticulate", repos = "https://cloud.r-project.org")
  }

  python <- .transglasso_find_python(python)
  if (length(work_dir) != 1L || !nzchar(work_dir) || !dir.exists(work_dir)) {
    stop("`work_dir` must point to the directory containing the Python solvers.", call. = FALSE)
  }
  # Avoid normalizePath() here: some Windows R installations corrupt paths
  # containing non-ASCII characters when the startup locale is misconfigured.
  work_dir <- path.expand(work_dir)
  required_files <- c("transglasso.py", "transmtglasso.py", "dtrace.py")
  candidate_dirs <- unique(c(
    work_dir,
    file.path(work_dir, "Trans-Glasso-R"),
    getwd(),
    file.path(getwd(), "Trans-Glasso-R")
  ))
  has_solvers <- vapply(candidate_dirs, function(path) {
    dir.exists(path) && all(file.exists(file.path(path, required_files)))
  }, logical(1))
  if (any(has_solvers)) {
    work_dir <- candidate_dirs[which(has_solvers)[1L]]
  } else {
    missing_files <- required_files[!file.exists(file.path(work_dir, required_files))]
    stop(
      sprintf(
        paste0(
          "Missing Python solver file(s) in `%s`: %s. ",
          "Set `work_dir` to the folder containing transglasso.py."
        ),
        work_dir,
        paste(missing_files, collapse = ", ")
      ),
      call. = FALSE
    )
  }

  if (!.transglasso_python_packages_ok(python)) {
    if (!isTRUE(install)) {
      stop(
        paste0(
          "Python packages numpy, scipy and scikit-learn are required. ",
          "Run setup_transglasso_python(install = TRUE, python = ...)."
        ),
        call. = FALSE
      )
    }
    install_output <- system2(
      python,
      c("-m", "pip", "install", "numpy", "scipy", "scikit-learn"),
      stdout = TRUE,
      stderr = TRUE
    )
    install_status <- attr(install_output, "status")
    if (!is.null(install_status) && install_status != 0L) {
      stop(
        paste(c("Python dependency installation failed:", install_output), collapse = "\n"),
        call. = FALSE
      )
    }
  }

  key <- paste(python, work_dir, sep = "|")
  if (!is.null(.transglasso_state$modules) && identical(.transglasso_state$key, key)) {
    return(invisible(.transglasso_state$modules$config))
  }

  reticulate::use_python(python, required = TRUE)
  py_os <- reticulate::import("os", convert = TRUE)
  py_sys <- reticulate::import("sys", convert = FALSE)
  py_work_dir <- as.character(py_os$path$abspath(work_dir))
  py_sys$path$insert(0L, py_work_dir)
  modules <- list(
    dtrace = reticulate::import("dtrace", convert = TRUE),
    transmtglasso = reticulate::import("transmtglasso", convert = TRUE),
    transglasso = reticulate::import("transglasso", convert = TRUE),
    sklearn_cov = reticulate::import("sklearn.covariance", convert = TRUE)
  )
  modules$config <- list(
    python = python,
    work_dir = py_work_dir,
    numpy = as.character(reticulate::import("numpy")$`__version__`),
    sklearn = as.character(reticulate::import("sklearn")$`__version__`)
  )
  .transglasso_state$modules <- modules
  .transglasso_state$key <- key
  invisible(modules$config)
}

.transglasso_as_matrix <- function(x, label) {
  if (is.data.frame(x) && !all(vapply(x, is.numeric, logical(1)))) {
    stop(sprintf("`%s` contains non-numeric columns.", label), call. = FALSE)
  }
  x <- as.matrix(x)
  if (!is.numeric(x) || length(dim(x)) != 2L) {
    stop(sprintf("`%s` must be a numeric matrix or data.frame.", label), call. = FALSE)
  }
  storage.mode(x) <- "double"
  if (nrow(x) < 3L) {
    stop(sprintf("`%s` must contain at least 3 observations.", label), call. = FALSE)
  }
  if (ncol(x) < 2L) {
    stop(sprintf("`%s` must contain at least 2 variables.", label), call. = FALSE)
  }
  if (any(!is.finite(x))) {
    stop(sprintf("`%s` contains NA, NaN or infinite values.", label), call. = FALSE)
  }
  x
}

.transglasso_prepare_data <- function(target_data, source_data, center) {
  target <- .transglasso_as_matrix(target_data, "target_data")
  if (is.matrix(source_data) || is.data.frame(source_data)) {
    source_data <- list(source_data)
  }
  if (!is.list(source_data) || !length(source_data)) {
    stop("`source_data` must be a non-empty list of matrices/data.frames.", call. = FALSE)
  }

  source_names <- names(source_data)
  if (is.null(source_names)) {
    source_names <- paste0("source_", seq_along(source_data))
  } else {
    empty <- is.na(source_names) | !nzchar(source_names)
    source_names[empty] <- paste0("source_", which(empty))
    source_names <- make.unique(source_names)
  }

  target_colnames <- colnames(target)
  if (!is.null(target_colnames) && anyDuplicated(target_colnames)) {
    stop("`target_data` column names must be unique.", call. = FALSE)
  }

  sources <- lapply(seq_along(source_data), function(k) {
    x <- .transglasso_as_matrix(source_data[[k]], sprintf("source_data[[%d]]", k))
    if (ncol(x) != ncol(target)) {
      stop(
        sprintf(
          "source_data[[%d]] has %d columns; target_data has %d.",
          k, ncol(x), ncol(target)
        ),
        call. = FALSE
      )
    }
    if (!is.null(colnames(x)) && anyDuplicated(colnames(x))) {
      stop(sprintf("source_data[[%d]] column names must be unique.", k), call. = FALSE)
    }
    if (!is.null(target_colnames) && !is.null(colnames(x))) {
      if (!setequal(target_colnames, colnames(x))) {
        stop(sprintf("source_data[[%d]] has different variables.", k), call. = FALSE)
      }
      x <- x[, target_colnames, drop = FALSE]
    }
    x
  })
  names(sources) <- source_names

  all_data <- c(list(target_data = target), sources)
  constant_columns <- lapply(all_data, function(x) {
    which(apply(x, 2L, stats::sd) <= sqrt(.Machine$double.eps))
  })
  bad <- which(lengths(constant_columns) > 0L)
  if (length(bad)) {
    labels <- names(all_data)[bad]
    stop(
      sprintf("Constant or near-constant variables found in: %s.", paste(labels, collapse = ", ")),
      call. = FALSE
    )
  }

  target_center <- colMeans(target)
  source_centers <- lapply(sources, colMeans)
  if (isTRUE(center)) {
    target <- sweep(target, 2L, target_center, FUN = "-")
    sources <- Map(function(x, mu) sweep(x, 2L, mu, FUN = "-"), sources, source_centers)
  }
  names(sources) <- source_names

  list(
    target = target,
    sources = sources,
    source_names = source_names,
    variable_names = target_colnames,
    centers = list(target = target_center, sources = source_centers)
  )
}

.transglasso_project_spd <- function(omega, min_eigenvalue) {
  omega <- 0.5 * (omega + t(omega))
  eig <- eigen(omega, symmetric = TRUE)
  values <- pmax(eig$values, min_eigenvalue)
  projected <- sweep(eig$vectors, 2L, values, FUN = "*") %*% t(eig$vectors)
  0.5 * (projected + t(projected))
}

.transglasso_prediction_error <- function(omega, validation_data, min_eigenvalue) {
  omega <- .transglasso_project_spd(omega, min_eigenvalue)
  p <- ncol(validation_data)
  empirical_cov <- crossprod(validation_data) / nrow(validation_data)
  log_det <- sum(log(eigen(omega, symmetric = TRUE, only.values = TRUE)$values))
  (sum(empirical_cov * t(omega)) - log_det) / (2 * p) + 0.5 * log(2 * pi)
}

.transglasso_dtrace_fit <- function(
    target, source, modules, dtrace_multipliers, sparsity_tol,
    eps_abs, eps_rel, max_iter_dtrace, verbose) {
  p <- ncol(target)
  penalty_grid <- dtrace_multipliers * sqrt(log(p) / min(nrow(target), nrow(source)))
  model <- modules$dtrace$DTraceBIC(
    target_data = target,
    source_data = source,
    penal_param_list = as.numeric(penalty_grid),
    eps_abs = eps_abs,
    eps_rel = eps_rel,
    max_iter = as.integer(max_iter_dtrace)
  )
  model$bic(print_info = isTRUE(verbose))
  difference <- as.matrix(model$best_model)
  list(
    difference = difference,
    sparsity = sum(abs(difference) > sparsity_tol),
    penalty = as.numeric(model$penal_param_bic),
    bic = as.numeric(model$bic_error)
  )
}

.transglasso_glasso_fit <- function(
    target, modules, glasso_cv_folds, max_iter_glasso, min_eigenvalue) {
  inner_folds <- max(2L, min(as.integer(glasso_cv_folds), nrow(target)))
  model <- modules$sklearn_cov$GraphicalLassoCV(
    cv = as.integer(inner_folds),
    assume_centered = TRUE,
    max_iter = as.integer(max_iter_glasso)
  )
  model$fit(target)
  precision <- .transglasso_project_spd(as.matrix(model$precision_), min_eigenvalue)
  list(
    precision = precision,
    precision_mt = NULL,
    model = model,
    lambda_transmtglasso = as.numeric(model$alpha_)
  )
}

.transglasso_known_fit <- function(
    target, sources, source_indices, difference_networks, modules,
    dtrace_multipliers, transmtglasso_multipliers, penal_param_admm,
    eps_abs, eps_rel, max_iter_dtrace, max_iter_transmtglasso,
    max_iter_glasso, glasso_cv_folds, min_eigenvalue, verbose) {
  if (!length(source_indices)) {
    return(.transglasso_glasso_fit(
      target, modules, glasso_cv_folds, max_iter_glasso, min_eigenvalue
    ))
  }

  chosen_sources <- unname(sources[source_indices])
  chosen_differences <- unname(difference_networks[source_indices])
  p <- ncol(target)
  source_sizes <- vapply(chosen_sources, nrow, integer(1))
  dtrace_grid <- dtrace_multipliers *
    sqrt(log(p) / min(c(nrow(target), source_sizes)))
  transmtglasso_grid <- transmtglasso_multipliers *
    sqrt(log(p) / (nrow(target) + sum(source_sizes)))

  model <- modules$transglasso$TransGLassoMS(
    target_data = target,
    source_data = chosen_sources,
    diff_network_list = chosen_differences,
    penal_param_dtrace_list = as.numeric(dtrace_grid),
    penal_param_transmtglasso_list = as.numeric(transmtglasso_grid),
    penal_param_admm = penal_param_admm,
    eps_abs = eps_abs,
    eps_rel = eps_rel,
    max_iter_dtrace = as.integer(max_iter_dtrace),
    max_iter_transmtglasso = as.integer(max_iter_transmtglasso)
  )
  model$model_selction(method = "BIC", print_info = isTRUE(verbose))
  precision <- .transglasso_project_spd(
    as.matrix(model$precision_matrix_target), min_eigenvalue
  )
  list(
    precision = precision,
    precision_mt = model$precision_matrices_mt,
    model = model,
    lambda_transmtglasso = as.numeric(model$chosen_penal_param_transmtglasso)
  )
}

# Unknown-informative-set Trans-Glasso. Only target_data and source_data are
# required. The remaining parameters control CV, tuning grids and numerics.
transglasso_unknownA <- function(
    target_data,
    source_data,
    n_folds = 5L,
    seed = 1L,
    center = TRUE,
    dtrace_multipliers = c(10, 5, 1, 0.5, 0.1, 0.05, 0.01),
    transmtglasso_multipliers = c(10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005),
    sparsity_tol = 0,
    min_eigenvalue = 1e-3,
    penal_param_admm = 1,
    eps_abs = 1e-4,
    eps_rel = 1e-4,
    max_iter_dtrace = 500L,
    max_iter_transmtglasso = 5000L,
    max_iter_glasso = 100L,
    glasso_cv_folds = 5L,
    python = NULL,
    install_dependencies = FALSE,
    work_dir = .transglasso_default_work_dir,
    verbose = TRUE,
    keep_python_model = FALSE) {
  call <- match.call()
  positive_grid <- function(x, label) {
    if (!is.numeric(x) || !length(x) || any(!is.finite(x)) || any(x <= 0)) {
      stop(sprintf("`%s` must contain positive finite numbers.", label), call. = FALSE)
    }
    sort(unique(as.numeric(x)), decreasing = TRUE)
  }
  dtrace_multipliers <- positive_grid(dtrace_multipliers, "dtrace_multipliers")
  transmtglasso_multipliers <- positive_grid(
    transmtglasso_multipliers, "transmtglasso_multipliers"
  )
  if (!is.numeric(sparsity_tol) || length(sparsity_tol) != 1L ||
      !is.finite(sparsity_tol) || sparsity_tol < 0) {
    stop("`sparsity_tol` must be one non-negative finite number.", call. = FALSE)
  }
  if (!is.numeric(min_eigenvalue) || length(min_eigenvalue) != 1L ||
      !is.finite(min_eigenvalue) || min_eigenvalue <= 0) {
    stop("`min_eigenvalue` must be one positive finite number.", call. = FALSE)
  }
  positive_scalar <- function(x, label) {
    if (!is.numeric(x) || length(x) != 1L || !is.finite(x) || x <= 0) {
      stop(sprintf("`%s` must be one positive finite number.", label), call. = FALSE)
    }
  }
  positive_scalar(penal_param_admm, "penal_param_admm")
  positive_scalar(eps_abs, "eps_abs")
  positive_scalar(eps_rel, "eps_rel")
  iteration_values <- c(
    max_iter_dtrace = max_iter_dtrace,
    max_iter_transmtglasso = max_iter_transmtglasso,
    max_iter_glasso = max_iter_glasso
  )
  if (any(!is.finite(iteration_values)) || any(iteration_values < 1) ||
      any(iteration_values != as.integer(iteration_values))) {
    stop("All `max_iter_*` parameters must be positive integers.", call. = FALSE)
  }
  if (!is.numeric(glasso_cv_folds) || length(glasso_cv_folds) != 1L ||
      !is.finite(glasso_cv_folds) || glasso_cv_folds < 2 ||
      glasso_cv_folds != as.integer(glasso_cv_folds)) {
    stop("`glasso_cv_folds` must be an integer of at least 2.", call. = FALSE)
  }
  if (!is.numeric(seed) || length(seed) != 1L || !is.finite(seed)) {
    stop("`seed` must be one finite number.", call. = FALSE)
  }

  prepared <- .transglasso_prepare_data(target_data, source_data, center)
  target <- prepared$target
  sources <- prepared$sources
  n_sources <- length(sources)
  n_folds <- as.integer(n_folds)
  if (length(n_folds) != 1L || is.na(n_folds) || n_folds < 2L ||
      n_folds > floor(nrow(target) / 2L)) {
    stop(
      "`n_folds` must be at least 2 and leave at least 2 validation observations per fold.",
      call. = FALSE
    )
  }

  setup_transglasso_python(
    python = python,
    install = install_dependencies,
    work_dir = work_dir
  )
  modules <- .transglasso_state$modules

  if (isTRUE(verbose)) {
    message("Step 1/3: ranking all sources with BIC-selected D-Trace networks")
  }
  ranking_fits <- lapply(seq_len(n_sources), function(k) {
    if (isTRUE(verbose)) {
      message(sprintf("  Source %d/%d: %s", k, n_sources, prepared$source_names[[k]]))
    }
    .transglasso_dtrace_fit(
      target, sources[[k]], modules, dtrace_multipliers, sparsity_tol,
      eps_abs, eps_rel, max_iter_dtrace, verbose
    )
  })
  difference_networks <- lapply(ranking_fits, `[[`, "difference")
  names(difference_networks) <- prepared$source_names
  sparsity_scores <- vapply(ranking_fits, `[[`, numeric(1), "sparsity")
  dtrace_penalties <- vapply(ranking_fits, `[[`, numeric(1), "penalty")
  source_order <- order(sparsity_scores, seq_len(n_sources))

  if (isTRUE(verbose)) {
    message(
      "Source order (most to least informative): ",
      paste(prepared$source_names[source_order], collapse = " -> ")
    )
    message("Step 2/3: selecting the number of sources by target-only cross-validation")
  }

  set.seed(as.integer(seed))
  fold_id <- sample(rep(seq_len(n_folds), length.out = nrow(target)))
  cv_error <- matrix(
    NA_real_, nrow = n_sources + 1L, ncol = n_folds,
    dimnames = list(paste0("K=", 0:n_sources), paste0("Fold", seq_len(n_folds)))
  )

  for (fold in seq_len(n_folds)) {
    if (isTRUE(verbose)) message(sprintf("  Outer fold %d/%d", fold, n_folds))
    validation_index <- which(fold_id == fold)
    training_index <- which(fold_id != fold)
    target_train <- target[training_index, , drop = FALSE]
    target_validation <- target[validation_index, , drop = FALSE]
    if (isTRUE(center)) {
      fold_center <- colMeans(target_train)
      target_train <- sweep(target_train, 2L, fold_center, FUN = "-")
      target_validation <- sweep(target_validation, 2L, fold_center, FUN = "-")
    }

    fold_differences <- vector("list", n_sources)
    for (k in seq_len(n_sources)) {
      fold_differences[[k]] <- .transglasso_dtrace_fit(
        target_train, sources[[k]], modules, dtrace_multipliers, sparsity_tol,
        eps_abs, eps_rel, max_iter_dtrace, FALSE
      )$difference
    }

    for (k_chosen in 0:n_sources) {
      chosen <- if (k_chosen == 0L) integer(0) else source_order[seq_len(k_chosen)]
      fit <- .transglasso_known_fit(
        target_train, sources, chosen, fold_differences, modules,
        dtrace_multipliers, transmtglasso_multipliers, penal_param_admm,
        eps_abs, eps_rel, max_iter_dtrace, max_iter_transmtglasso,
        max_iter_glasso, glasso_cv_folds, min_eigenvalue, FALSE
      )
      cv_error[k_chosen + 1L, fold] <- .transglasso_prediction_error(
        fit$precision, target_validation, min_eigenvalue
      )
    }
  }

  cv_mean <- rowMeans(cv_error)
  k_best <- which.min(cv_mean) - 1L
  selected_sources <- if (k_best == 0L) integer(0) else source_order[seq_len(k_best)]

  if (isTRUE(verbose)) {
    message(sprintf("Selected %d informative source(s).", k_best))
    message("Step 3/3: refitting the final estimator with all target observations")
  }
  final_fit <- .transglasso_known_fit(
    target, sources, selected_sources, difference_networks, modules,
    dtrace_multipliers, transmtglasso_multipliers, penal_param_admm,
    eps_abs, eps_rel, max_iter_dtrace, max_iter_transmtglasso,
    max_iter_glasso, glasso_cv_folds, min_eigenvalue, verbose
  )

  result <- list(
    Ome_hat_trans_glasso = final_fit$precision,
    KA_best = k_best,
    K_selected = k_best,
    selected_sources = selected_sources,
    selected_source_names = prepared$source_names[selected_sources],
    source_order = source_order,
    source_order_names = prepared$source_names[source_order],
    source_sparsity_scores = sparsity_scores,
    dtrace_penalties = dtrace_penalties,
    cv_error = cv_error,
    cv_mean = cv_mean,
    precision_target = final_fit$precision,
    python_config = modules$config,
    call = call
  )
  if (isTRUE(keep_python_model)) result$python_model <- final_fit$model
  class(result) <- "transglasso_unknownA"
  result
}

# Backward-compatible alias using snake_case.
transglasso_unknown_a <- transglasso_unknownA

print.transglasso_unknownA <- function(x, ...) {
  cat("Trans-Glasso-CV (unknown informative set)\n")
  cat("  Selected sources:", x$K_selected, "\n")
  if (x$K_selected > 0L) {
    cat("  Source names:", paste(x$selected_source_names, collapse = ", "), "\n")
  }
  cat("  Minimum mean CV error:", format(min(x$cv_mean), digits = 6L), "\n")
  cat(
    "  Precision matrix dimension:",
    nrow(x$precision_target), "x", ncol(x$precision_target), "\n"
  )
  invisible(x)
}
