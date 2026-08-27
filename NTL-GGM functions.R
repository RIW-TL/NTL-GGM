library(glmnet)
library(ncvreg)
library(bayesdistreg)
library(flare)
library(foreach)
library(doParallel)
#--------------------------------------------------

# compute the likelihood loss
Dist = function(Ome_hat,X.test){
  
  p = dim(Ome_hat)[1]
  eigens = eigen(Ome_hat)$values
  value = sum(diag(cov(X.test) %*% Ome_hat))/(2*p) - 
    sum(log(eigens[eigens > 0]))/(2*p)
  return(value)
}

# rcv variance estimation 
Est_var = function(X,Y){
  
  n0 = nrow(X)
  # the first half
  id = 1:(n0/2)
  fit = cv.glmnet(X[id,],Y[id],alpha = 1)
  sub = which(coef(fit)[-1] != 0)
  
  X_sub = X[-id,sub]
  Y_sub = Y[-id]
  beta_sub = coef(fit)[-1][sub]
  # P = X_sub %*% solve(t(X_sub) %*% X_sub) %*% t(X_sub)
  var_hat_1 = sum((Y_sub - X_sub %*% matrix(beta_sub))^2) /(n0/2 - length(sub))
  
  
  # the second half
  id = ((n0/2) + 1):n0
  fit = cv.glmnet(X[-id,],Y[-id],alpha = 1)
  sub = which(coef(fit)[-1] != 0)
  
  X_sub = X[id,sub]
  Y_sub = Y[id]
  beta_sub = coef(fit)[-1][sub]
  # P = X_sub %*% solve(t(X_sub) %*% X_sub) %*% t(X_sub)
  var_hat_2 = sum((Y_sub - X_sub %*% matrix(beta_sub))^2) /(n0/2 - length(sub))
  
  var_hat = (var_hat_1 + var_hat_2)/2
  
 return(var_hat)
}

## Node-wise (target-only)
fit.neighbor = function(Xt, penalty = c("lasso", "SCAD"),
                        ncores = max(1, parallel::detectCores() - 2)) {
  
  penalty = match.arg(penalty)
  
  n0 = nrow(Xt)
  p  = ncol(Xt)
  
  # 建立并行集群
  cl = parallel::makeCluster(ncores)
  doParallel::registerDoParallel(cl)
  
  # 保证函数结束时关闭集群
  on.exit(parallel::stopCluster(cl))
  
  Ome_hat = foreach::foreach(
    j = 1:p,
    .combine = cbind,
    .packages = c("glmnet", "ncvreg")
  ) %dopar% {
    
    if (penalty == "lasso") {
      
      fit = glmnet::cv.glmnet(
        Xt[, -j, drop = FALSE],
        Xt[, j],
        family = "gaussian",
        alpha = 1
      )
      
      beta.hat = as.numeric(coef(fit)[-1])
      
    } else {
      
      fit = ncvreg::cv.ncvreg(
        Xt[, -j, drop = FALSE],
        Xt[, j],
        family = "gaussian",
        penalty = "SCAD"
      )
      
      beta.hat = as.numeric(coef(fit)[-1])
    }
    
    residual = Xt[, j] - Xt[, -j, drop = FALSE] %*% beta.hat
    
    var = sum(residual^2) /
      (n0 - sum(beta.hat != 0))
    
    # 第 j 列
    omega_j = numeric(p)
    omega_j[j]  = 1 / var
    omega_j[-j] = -omega_j[j] * beta.hat
    
    omega_j
  }
  
  Ome.hat = sym_def(Ome_hat)
  
  return(Ome.hat)
}

# RIW-TL in each column
fit.RIW = function(Xt,Yt,XS,YS,
                   c_A1 = 0.6,c_M0 = 1,c_M1 = 1,
                   Ome_t_initial,Ome_s_initial,j){
  
  p = ncol(Xt)
  K = length(XS)
  
  # KDE and weighted sources data
  #-------------------------------------
  Ydata_star = Xdata_star = eta = list()
  error_0 = error_1 = list()
  
  for (k in 1:K) {
    
    beta0.initial.hat = Ome_t_initial[-j,j]
    beta1.initial.hat = Ome_s_initial[[k]][-j,j]
    
    error_0[[k]] = YS[[k]] - XS[[k]] %*% beta0.initial.hat
    error_1[[k]] = YS[[k]] - XS[[k]] %*% beta1.initial.hat
    
    # the estimation of importance weights
    fenzi = dnorm(error_0[[k]],0,sqrt(Ome_s_initial[[k]][j,j]))
    fenmu = dnorm(error_1[[k]],0,sqrt(Ome_s_initial[[k]][j,j]))
    weight = fenzi/fenmu
    
    Ydata_star[[k]] = sqrt(weight)*YS[[k]]
    Xdata_star_matrix = matrix(0,nrow(XS[[k]]),p)
    for (i in 1:nrow(XS[[k]])) {
      
      Xdata_star_matrix[i,] = sqrt(weight[i])*XS[[k]][i,]
    }
    Xdata_star[[k]] = Xdata_star_matrix
    
    # contrast vectors adjusted by predictors
    eta[[k]] = XS[[k]] %*% (beta1.initial.hat - beta0.initial.hat)
  }
  
  
  # sample selection
  #-------------------------------------
  YY = Yt;XX = Xt
  for (k in 1:K) {
    
    A = quantile(abs(error_1[[k]]),c_A1)
    M = c_M0/(c_M1 + sum(abs(beta1.initial.hat - beta0.initial.hat)))
    
    id1 = which((abs(error_1[[k]]) <= A) & (abs(eta[[k]]) <= M))
    YY = c(YY,Ydata_star[[k]][id1])
    XX = rbind(XX,Xdata_star[[k]][id1,])
  }
  
  # weighted lasso optimization
  const = 0.5
  fit = glmnet(XX,YY,family = "gaussian",alpha = 1,
               lambda = const*sqrt(log(p)/length(YY)))
  
  beta_RIW_TL = coef(fit)[-1]
  
  return(beta_RIW_TL)
}

## Node-wise (our)
fit.NTL <- function(
    Xt, XS,
    c_A1 = 0.6, c_M0 = 1, c_M1 = 1,
    c_A2 = 0.6, c_M2 = 0.5,
    Ome_t_initial, Ome_s_initial,
    ncores = max(1, parallel::detectCores() - 2)) {
  
  n0 <- nrow(Xt)
  p  <- ncol(Xt)
  K  <- length(XS)
  
  # Data transformation
  Xtt <- Xt
  XSS <- XS
  
  for (j in 1:p) {
    Xtt[, j] <- Xt[, j] * (-1) * Ome_t_initial[j, j]
  }
  
  for (k in 1:K) {
    for (j in 1:p) {
      XSS[[k]][, j] <-
        XS[[k]][, j] * (-1) * Ome_s_initial[[k]][j, j]
    }
  }
  
  #------------------------------------------------------------
  
  # 每个并行任务估计精度矩阵的第 j 列
  fit_one_node <- function(j) {
    
    Y_tar <- Xtt[, j]
    X_tar <- Xt[, -j]
    
    Y_sou <- X_sou <- list()
    
    for (k in 1:K) {
      Y_sou[[k]] <- XSS[[k]][, j]
      X_sou[[k]] <- XS[[k]][, -j]
    }
    
    # Initial estimators
    beta0.initial.hat <- Ome_t_initial[-j, j]
    beta1.initial.hat <- matrix(0, p - 1, K)
    
    for (k in 1:K) {
      beta1.initial.hat[, k] <- Ome_s_initial[[k]][-j, j]
    }
    
    #===========================================================
    # Non-diagonal elements estimation
    #===========================================================
    
    omega_j <- numeric(p)
    
    omega_j[-j] <- fit.RIW(
      X_tar, Y_tar, X_sou, Y_sou,
      c_A1, c_M0, c_M1,
      Ome_t_initial, Ome_s_initial, j
    )
    
    #===========================================================
    # Diagonal element estimation
    #===========================================================
    
    # Sources selection
    Ome_s_diag <- rep(0, K)
    
    for (k in 1:K) {
      Ome_s_diag[k] <- Ome_s_initial[[k]][j, j]
    }
    
    eta <- Ome_t_initial[j, j] - Ome_s_diag
    info.id <- which(abs(eta) <= c_M2)
    
    if (length(info.id) == 0) {
      
      omega_j[j] <- Ome_t_initial[j, j]
      
    } else {
      
      # Estimation of diagonal elements
      RSS <- sum(
        (Xtt[, j] -
           Xt[, -j] %*% Ome_t_initial[-j, j])^2
      )
      
      n.var <- n0
      
      num.nonzero <- length(
        which(Ome_t_initial[-j, j] != 0)
      )
      
      for (k in info.id) {
        
        Y <- XSS[[k]][, j]
        X <- XS[[k]][, -j]
        
        # Estimation of importance weights
        error_1 <-
          Y - X %*% Ome_s_initial[[k]][-j, j]
        
        fenzi <- dnorm(
          error_1,
          0,
          sqrt(Ome_t_initial[j, j])
        )
        
        fenmu <- dnorm(
          error_1,
          0,
          sqrt(Ome_s_initial[[k]][j, j])
        )
        
        weight <- fenzi / fenmu
        
        # Sample selection
        A2 <- quantile(abs(error_1), c_A2)
        id <- which(abs(error_1) <= A2)
        
        n <- length(Y)
        y <- Y * sqrt(weight)
        
        for (i in 1:n) {
          X[i, ] <- X[i, ] * sqrt(weight[i])
        }
        
        y <- y[id]
        X <- X[id, ]
        
        RSS <- RSS + sum(
          (y -
             X %*% Ome_s_initial[[k]][-j, j])^2
        )
        
        n.var <- n.var + length(y)
        
        num.nonzero <- num.nonzero +
          length(
            which(Ome_s_initial[[k]][-j, j] != 0)
          )
      }
      
      omega_j[j] <- RSS / (n.var - num.nonzero)
    }
    
    # 返回精度矩阵的第 j 列
    omega_j
  }
  
  #=============================================================
  # Parallel estimation over j = 1, ..., p
  #=============================================================
  
  if (is.null(ncores)) {
    detected_cores <- parallel::detectCores(logical = TRUE)
    
    if (is.na(detected_cores)) {
      ncores <- 1L
    } else {
      ncores <- max(1L, detected_cores - 1L)
    }
  }
  
  ncores <- max(1L, min(as.integer(ncores), p))
  
  if (ncores == 1L) {
    
    # Serial calculation
    Ome_trans <- vapply(
      X = seq_len(p),
      FUN = fit_one_node,
      FUN.VALUE = numeric(p)
    )
    
  } else {
    
    if (!requireNamespace("foreach", quietly = TRUE)) {
      stop("Package `foreach` is required for parallel calculation.")
    }
    
    if (!requireNamespace("doParallel", quietly = TRUE)) {
      stop("Package `doParallel` is required for parallel calculation.")
    }
    
    cl <- parallel::makeCluster(ncores)
    
    on.exit({
      parallel::stopCluster(cl)
      foreach::registerDoSEQ()
    }, add = TRUE)
    
    doParallel::registerDoParallel(cl)
    
    iterator <- foreach::foreach(
      j = seq_len(p),
      .combine = cbind,
      .packages = "glmnet",
      .export = "fit.RIW"
    )
    
    Ome_trans <- foreach::`%dopar%`(
      iterator,
      fit_one_node(j)
    )
  }
  
  Ome_trans <- sym_def(Ome_trans)
  
  return(Ome_trans)
}


