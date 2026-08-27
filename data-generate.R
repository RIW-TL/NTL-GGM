library(glmnet)
library(ncvreg)
library(mvtnorm)
library(bayesdistreg)
library(fastclime)
library(flare)
library(lavaSearch2)
#--------------------------------------------------

# symmetry and positive definite
sym_def = function(Omega){
  
  Omega = (Omega + t(Omega))/2
  eig_val = eigen(Omega)$values
  
  if(min(eig_val) >= 0){
    
    Omega = Omega
  }else{
    
    c = 0.1 - min(eig_val)
    Omega = Omega + c*diag(1,p)
  }
  
  return(Omega)
}

generate_block_local_setting <- function(
    p,K,
    block_size = 20,
    sparse_prob = 0.2,
    small_noise = 0.05,
    large_noise_lower = 0.5,
    large_noise_upper = 1.0,
    min_eigen = 0.1,
    seed = 1
) {
  set.seed(seed)
  
  num_blocks <- p / block_size
  stopifnot(num_blocks == K)
  
  make_target_block <- function(block_size) {
    B <- matrix(0, block_size, block_size)
    
    for (i in 1:block_size) {
      for (j in 1:block_size) {
        if (i == j) {
          B[i, j] <- 1.5
        } else if (abs(i - j) <= 4) {
          B[i, j] <- 0.6 ^ abs(i - j)
        }
      }
    }
    
    B
  }
  
  make_pd_block <- function(B, min_eigen = 0.1) {
    lambda_min <- min(eigen(B, symmetric = TRUE, only.values = TRUE)$values)
    
    if (lambda_min < min_eigen) {
      B <- B + diag(min_eigen - lambda_min, nrow(B))
    }
    
    B
  }
  
  make_sparse_noise <- function(
    block_size,
    prob,
    lower,
    upper,
    allow_negative = FALSE
  ) {
    M <- matrix(0, block_size, block_size)
    
    for (i in 1:(block_size - 1)) {
      for (j in (i + 1):block_size) {
        if (runif(1) < prob) {
          val <- runif(1, lower, upper)
          
          if (allow_negative) {
            val <- sample(c(-1, 1), 1) * val
          }
          
          M[i, j] <- val
          M[j, i] <- val
        }
      }
    }
    
    diag(M) <- 0
    M
  }
  
  B_target_list <- lapply(1:num_blocks, function(b) {
    make_pd_block(make_target_block(block_size), min_eigen)
  })
  
  Omega_target <- as.matrix(bdiag(B_target_list))
  
  Omega_sources <- vector("list", K)
  
  for (k in 1:K) {
    B_source_list <- vector("list", num_blocks)
    
    for (b in 1:num_blocks) {
      B0 <- B_target_list[[b]]
      
      if (b == k) {
        # Similar block: sparse small perturbation
        E <- make_sparse_noise(
          block_size = block_size,
          prob = sparse_prob,
          lower = 0,
          upper = small_noise,
          allow_negative = TRUE
        )
        
        Bk <- B0 + E
        
        # Safety check only; usually no correction needed
        Bk <- make_pd_block(Bk, min_eigen)
        
      } else {
        # Dissimilar block: sparse large perturbation
        H <- make_sparse_noise(
          block_size = block_size,
          prob = sparse_prob,
          lower = large_noise_lower,
          upper = large_noise_upper,
          allow_negative = TRUE
        )
        
        Bk <- B0 + H
        
        # Block-specific PD correction only for dissimilar block
        Bk <- make_pd_block(Bk, min_eigen)
      }
      
      B_source_list[[b]] <- Bk
    }
    
    Omega_sources[[k]] <- as.matrix(bdiag(B_source_list))
  }
  
  list(
    Omega_target = Omega_target,
    Omega_sources = Omega_sources
  )
}

## parameter generation
para_fun = function(p,K,k0,d,s = 5,
                    case = c("band","random","block"),
                    seed = NULL) {
  
  if (!is.null(seed)) {
    set.seed(seed)
  }
  
  
  if(case == "block"){
  
  fit.block = generate_block_local_setting(p,K,seed = seed)
  Omega_0 = fit.block$Omega_target
  Omega_s = fit.block$Omega_sources
  }else{
    
    # target precision matrix
    #-------------------------------------
    if(case == "band"){
      
      Omega_0 = matrix(0,p,p)
      
      for (i in 1:p) {
        for (j in 1:p) {
          
          if (abs(i - j) <= 4) {
            Omega_0[i, j] <- 0.6^abs(i - j)
          }
          
        }
      }
    }else{
      
      if(case == "random"){
        
        Omega_0 = matrix(runif(p*p, 0.3, 0.5), nrow = p, ncol = p)
        diag(Omega_0) = rep(1,p)
        
        Omega_sym = (Omega_0 + t(Omega_0)) / 2
        
        # remain the first s maximum absolute values
        Omega_threshold <- Omega_sym
        
        # thresholding in row
        for (i in 1:p) {
          
          row_vals <- Omega_sym[i, ]
          top_s_idx <- order(abs(row_vals), decreasing = TRUE)[1:s]
          row_vals[-top_s_idx] <- 0
          Omega_threshold[i, ] <- row_vals
        }
        
        # thresholding in column
        for (j in 1:p) {
          
          col_vals <- Omega_threshold[, j]
          top_s_idx <- order(abs(col_vals), decreasing = TRUE)[1:s]
          col_vals[-top_s_idx] <- 0
          Omega_threshold[, j] <- col_vals
        }
        
        Omega_0 = Omega_threshold
        
        Omega_0 = sym_def(Omega_0)
      }
    }
 
    
    # source precision matrices
    #--------------------------------------
    Omega_s = list()
    for (k in 1:K) {
      
      Gamma = matrix(0, nrow = p, ncol = p)
      
      if(k <= k0){
        
        for (j in 1:p) {
          
          Gamma[j, j] = 0
          Gamma[-j, j] = sample(c(0,d/p),p - 1,replace = TRUE,prob = c(0.95,0.05))
        }
      }else{
        
        for (j in 1:p) {
          
          Gamma[j, j] = 2
          Gamma[-j, j] = sample(c(0,0.5),p - 1,replace = TRUE,prob = c(0.7,0.3))
        }
      }  
      
      Omega_s[[k]] = Omega_0 + Gamma
      Omega_s[[k]] = sym_def(Omega_s[[k]])
    }
  }
  
  return(list(Omega_0 = Omega_0, Omega_s = Omega_s))
}

## data generation
Data_gen = function(n0,n1,p,K,k0,d,s,
                    case = c("band","random","block")){
  
  fit = para_fun(p,K,k0,d,s,case)
  Ome_t = fit$Omega_0
  Ome_s = fit$Omega_s
  
  # data generation
  Xt = rmvnorm(n0,rep(0,p),solve(Ome_t))
  XS = list()
  for (k in 1:K) {
    
    XS[[k]] = rmvnorm(n1,rep(0,p),solve(Ome_s[[k]]))
  }
  
  list = list(Ome_t = Ome_t,Ome_s = Ome_s,
              Xt = Xt,XS = XS)
}





