library(fastclime)
library(lavaSearch2) 
#---------------------------------------

## Trans-CLIME
Myfastclime.s = function(X, Bmat, lambda = 0.1, scale = T, n){
  
  p<-ncol(X)
  obj=rep(-1,2*p)
  obj_bar=rep(0,2*p)
  rhs_bar<-rep(1, 2*p)
  if(isSymmetric(X, tol=10^(-4))){
    Sig.hat<-X
  }else{
    Sig.hat<-cov(X)
  }
  Sig.hat0<-Sig.hat
  feasible=T
  Theta.hat<-NULL
  Sig.diag<-diag(Sig.hat)
  Sig.hat<-cov2cor(Sig.hat)
  mat=rbind(cbind(Sig.hat,-Sig.hat),
            cbind(-Sig.hat,Sig.hat))
  for(j in 1:p){
    rhs <- c(Bmat[,j],-Bmat[,j])
    out.txt<-capture.output(  fastlp.re<-fastlp(obj=obj, mat=mat, rhs=rhs+rhs_bar*lambda))
    if(!grepl("optimal", out.txt) ){
      feasible=F
      break
    }
    Theta.hat<-cbind(Theta.hat,(fastlp.re[1:p]-fastlp.re[-(1:p)])/sqrt(Sig.diag[j])/sqrt(Sig.diag))
    if(scale & sum(Theta.hat[,j]==0)==p){
      feasible=F
      break
    }else if(scale){
      Theta.hat[,j]<-  as.numeric(Bmat[j,j]/ (Sig.hat0[j,]%*%Theta.hat[,j]))*Theta.hat[,j]
      # Theta.hat[,j]<-as.numeric(Theta.hat[j,j]/ (t(Theta.hat[,j])%*%Sig.hat0%*%Theta.hat[,j]))*Theta.hat[,j]
      
    }
  }
  if(!feasible){
    cat('Theta.hat not found','\n')
    Theta.hat<-solve(cov(X)+diag(lambda,p))%*%Bmat
  }
  list(Theta.hat=Theta.hat, conv=feasible)
}

Agg = function(Theta.init, X.til){
  p<-ncol(X.til)
  n.til<-nrow(X.til)
  v.mat<-sapply(1:p, function(j){
    W.j<-cov(X.til%*%cbind(Theta.init[,j], Theta.init[,p+j]))
    v0=rep(0,2)
    v0[which.min(c(W.j[1,1]-2*Theta.init[j,j], W.j[2,2]-2*Theta.init[j,p+j]))]<-1
    v0
  })
  Theta.hat<-sapply(1:p, function(j) cbind(Theta.init[,j], Theta.init[,p+j])%*% v.mat[,j])
  
  Theta.hat
}

Spd.proj = function(SigA.hat, eps=NULL){
  p=ncol(SigA.hat)
  if(is.null(eps)){
    eps<-5/p
  }
  feasible=1
  SigA.t<-SigA.hat
  if(min(eigen(SigA.t)$values) <=eps ){
    feasible=2
    SigA.t<-ADMM_proj(SigA.hat, epsilon=eps)$mat
  }
  SigA.t<-lavaSearch2:::symmetrize(SigA.t, update.upper = TRUE)
  list(mat=SigA.t, conv=feasible)
}

l1proj = function(v, b){
  
  stopifnot(b>0)
  
  u <- sort(abs(v),decreasing=TRUE)
  sv <- cumsum(u)
  rho <- max(which(u>(sv-b)/1:length(u)))
  theta <- max(0, (sv[rho]-b)/rho)
  w <-sign(v) * pmax(abs(v)-theta,0)
  
  return(w)
}

ADMM_proj = function(mat,
                     epsilon=1e-4,
                     mu=10,
                     it.max=1e3,
                     etol=1e-4,
                     etol_distance = 1e-4){
  
  
  
  p<-nrow(mat)
  
  # Initialization
  R<-diag(mat)
  S<-matrix(0,p,p)
  L<-matrix(0,p,p)
  
  itr<-0
  iteration <- eps_R <- eps_S <- eps_primal <- time <- distance <- NULL
  while (itr<it.max) {
    #print(itr)
    Rp<-R
    Sp<-S
    start <- Sys.time()
    # Subproblem I: R step
    W<-mat+S+mu*L
    W.eigdec<-eigen(W, symmetric=TRUE) 
    W.V<-W.eigdec$vectors
    W.D<-W.eigdec$values
    R<-W.V%*%diag(pmax(W.D,epsilon))%*%t(W.V)
    
    # Subproblem II: S step
    M<-R-mat-mu*L     
    S[lower.tri(S, diag = TRUE)]<-M[lower.tri(M, diag = TRUE)]-l1proj(v=M[lower.tri(M, diag = TRUE)],b=mu/2)    
    for (i in 2:p){
      for (j in 1:(i-1)){
        S[j,i]<-S[i,j]
      }
    }
    
    # L step: update the Lagrange parameter
    L<-L-(R-S-mat)/mu
    end <- Sys.time()
    #Stocking the values of different parameters with the number of iterations
    iteration <- c(iteration, itr)
    eps_R <- c(eps_R,max(abs(R-Rp)))
    eps_S <- c(eps_S,max(abs(S-Sp)))
    eps_primal <- c(eps_primal, max(abs(R-S-mat)))
    time <- c(time, end - start)
    distance <- c(distance,max(abs(R-mat)))
    
    # Stopping Rule                        
    #cat("check the stopping criterion:",max(abs(R-S-mat)),"\n")
    if (((max(abs(R-Rp))<etol) && (max(abs(S-Sp))<etol) && (max(abs(R-S-mat))<etol)) || (abs(max(abs(Rp-mat)) - max(abs(R-mat)))<etol_distance)){
      itr<-it.max
    } else {
      itr<-itr+1
    }
    
    if (itr%%20==0) {
      mu<-mu/2
    }
  }
  df_ADMM <- data.frame(iteration = iteration, eps_R = eps_R, eps_S=eps_S, eps_primal=eps_primal, time=time, distance=distance)
  return(list(mat=R,df_ADMM=df_ADMM))
  
}

Trans.CLIME = function(X, X.A, const, agg = T, X.til = NULL, Theta.cl){
  
  if(agg &is.null(X.til)){
    cat('no aggregation samples provided.','\n')
  }
  n0=nrow(X)
  nA<-nrow(X.A)
  p<-ncol(X)
  sigA.hat<-mean(apply(X.A, 2, sd))
  sig0.hat<-mean(apply(X, 2, sd))
  
  omega.l1<-mean(apply(Theta.cl,2, function(x) sum(abs(x))))
  Delta.re <- Myfastclime.s(X=diag(1,p), Bmat=diag(1,p)-t(Theta.cl)%*%cov(X.A), 
                            lambda = 2*omega.l1*sqrt(log(p)/n0), scale = F)
  if(Delta.re$conv){Delta.hat<-Delta.re$Theta.hat}else{ Delta.hat<-diag(0,p) }
  Theta.re <- Myfastclime.s(X=cov(X.A), Bmat=diag(1,p)-t(Delta.hat),
                            lambda = 2*const*sqrt(log(p)/nA))
  Theta.hat<-Theta.re$Theta.hat
  
  if(agg){
    
    Omega.hat <- Agg(Theta.init=cbind(Theta.cl, Theta.hat), X.til=X.til)
  }else{
    Omega.hat<-Theta.hat
  }
  Omega.hat
}
