library(mvtnorm)
library(Matrix)
library(fastclime)
library(lavaSearch2) 
##----------------------------------------------------------
# source("dara-generate.R")
# source("NTL-GGM functions.R")
# source("Trans-CLIME-functions.R")

# setwd("C:\\Users\\89261\\Nutstore\\1\\我的坚果云\\NTL-GGM-R codes\\Trans-Glasso-implement-by-R")
source("trans-glasso-implement.R")

load("use.tissue.RData")

##################################################################
## The third to fifth entries in use.tissue (list data) correspond to “Brain_Cortex,” “Brain_N.A.B.ganglia,” and “Brain_P.B.ganglia,” respectively. 
## Each is treated in turn as the target tissue, with the remaining K = 9 tissues serving as sources. 

## This file serves to compute the negative log-likelihood loss. 
#---------------------------------------------------

PE.loss = array(0,c(50,4,3))
for (l in 1:3) {
  
  colnames(PE.loss[,,l]) = c("Target-GGM","NTL-GGM","Trans-Glasso-CV","Trans-CLIME")
}

for (t in c(3:5)) {
  
  ############################################################
  # Take \(t = 3\) as an example.
  K = length(use.tissue) - 1
  # t = 3
  Xt = use.tissue[[t]] # Target data
  XS = use.tissue[-t]  # Source data
  ###########################################################
  
  for (ii in 1:50) {
    
    ##===========================================
    ## dividing data for training and testing data
    ##===========================================
    set.seed(ii)
    train.id = sample(1:length(Xt),prob = 0.7)
    Xt.train = Xt[train.id,]
    Xt.test = Xt[-train.id,]
    
    
    ##============================================================
    ## compute initial estimators used for NTL-GGM and Trans-CLIME
    ##============================================================
    Ome_hat_lasso = Ome_hat_SCAD = list()
    for (l in 1:(K + 1)) {
      
      Ome_hat_lasso[[l]] = fit.neighbor(use.tissue[[l]], penalty = "lasso")
      Ome_hat_SCAD[[l]] = fit.neighbor(use.tissue[[l]], penalty = "SCAD")
    }
    
    
    ##===========================================
    ## Target-GGM
    ##===========================================
    Ome_target_GGM = fit.neighbor(Xt.train, penalty = "lasso")
    PE.target.GGM = Dist(Ome_target_GGM,Xt.test)
    
    
    ##===========================================
    ## NTL-GGM
    ##===========================================
    
    Ome_NTL_GGM = fit.NTL(Xt.train,XS,
                          c_A1 = 0.8,c_M0 = 15,c_M1 = 0.5,
                          c_A2 = 0.8,c_M2 = 1,
                          Ome_t_initial = Ome_hat_lasso[[t]],
                          Ome_s_initial = Ome_hat_lasso[-t])
    PE.NTL.GGM = Dist(Ome_NTL_GGM,Xt.test)  
    
    
    ##===========================================
    ## Trans-CLIME
    ##===========================================
    n0 = nrow(Xt.train)
    p = ncol(Xt.train)
    
    X.A = c()
    n.vec = c(n0)
    for (k in 1:K) {
      
      X.A = rbind(X.A,XS[[k]])
      n.vec[k + 1] = nrow(XS[[k]])
    }
    
    n00 = round(n.vec[1]*4/5) 
    const = 0.5
    Theta.re0 = Myfastclime.s(X = Xt.train[1:n00,], Bmat = diag(1,p), 
                              lambda = const*2*sqrt(log(p)/n00))
    Theta.init = Theta.re0$Theta.hat
    Omega.tl1 = Trans.CLIME(X = Xt.train[1:n00,], X.A, const = const, 
                            X.til = Xt.train[(n00 + 1):n0,], Theta.cl = Theta.init)
    ind2 = (n0 - n00 + 1):n0
    Theta.re0 = Myfastclime.s(X = Xt.train[ind2,], Bmat = diag(1,p), 
                              lambda = const*2*sqrt(log(p)/length(ind2)))
    Theta.init = Theta.re0$Theta.hat
    Omega.tl2 = Trans.CLIME(X = Xt.train[ind2,], X.A, const = const,
                            X.til = Xt.train[1:(n0 - n00),], Theta.cl = Theta.init)
    Ome_hat_trans_clime = (Omega.tl1 + Omega.tl2)/2
    PE.trans.clime = Dist(Ome_hat_trans_clime,Xt.test)  
    
    
    ##===========================================
    ## Trans-CLIME
    ##===========================================  
    # fit.trans.glasso = transglasso_unknownA(Xt.train,XS)
    # Ome_hat_trans_glasso = fit.trans.glasso$Ome_hat_trans_glasso
    # PE.trans.glasso = Dist(Ome_hat_trans_glasso,Xt.test)    
    
    PE.loss[ii,,t - 2] = c(PE.target.GGM,PE.NTL.GGM,
                # PE.trans.glasso,
                PE.trans.clime)
  }
  
}







