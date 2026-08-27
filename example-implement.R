#----------------------------------------------
# source("data-generate.R")
# source("NTL-GGM functions.R")
# source("Trans-CLIME-functions.R")

setwd("C:\\Users\\89261\\Nutstore\\1\\我的坚果云\\NTL-GGM-R codes\\Trans-Glasso-implement-by-R")
source("trans-glasso-implement.R")
#----------------------------------------------

n0 = 200;p = 100;n1 = 1200;K = 5;s = 5;case = "band"
k0 = 3;d = 20

fit = Data_gen(n0,n1,p,K,k0,d,s,case)
Ome_t = fit$Ome_t;Ome_s = fit$Ome_s
Xt = fit$Xt;XS = fit$XS

##============================================
## Target-GGM
##============================================

Ome_hat = fit.neighbor(Xt,penalty = "lasso")
error.tar = norm(Ome_hat - Ome_t,"F")^2/p
error.diag.tar = sum(abs(diag(Ome_hat) - diag(Ome_t)))/p

##============================================
##NTL-GGM
##============================================

# const = 0.5
# Ome_t_initial = Myfastclime.s(X = Xt, Bmat = diag(1,p), 
#                                    lambda = const*2*sqrt(log(p)/n0))$Theta.hat

Ome_t_initial = fit.neighbor(Xt,penalty = "SCAD")

Ome_s_initial = list()
for (k in 1:K) {
  
  Ome_s_initial[[k]] = fit.neighbor(XS[[k]],penalty = "lasso")
}

Ome_trans = fit.NTL(Xt,XS,
                    c_A1 = 0.8,c_M0 = 15,c_M1 = 0.5,
                    c_A2 = 0.8,c_M2 = 1,
                    Ome_t_initial,
                    Ome_s_initial)
error.NTL = norm(Ome_trans - Ome_t,"F")^2/p;error.NTL
error.diag.NTL = sum(abs(diag(Ome_trans) - diag(Ome_t)))/p;error.diag.NTL


##============================================
##Trans-CLIME
##============================================

X.A = c()
for (k in 1:K) {X.A = rbind(X.A,XS[[k]])}

n.vec = c(n0,rep(n1,K))
n00 = round(n.vec[1]*4/5) 
const = 0.5
Theta.re0 = Myfastclime.s(X = Xt[1:n00,], Bmat = diag(1,p), 
                          lambda = const*2*sqrt(log(p)/n00))
Theta.init = Theta.re0$Theta.hat
Omega.tl1 = Trans.CLIME(X = Xt[1:n00,], X.A, const = const, 
                        X.til = Xt[(n00 + 1):n0,], Theta.cl = Theta.init)
ind2 = (n0 - n00 + 1):n0
Theta.re0 = Myfastclime.s(X = Xt[ind2,], Bmat = diag(1,p), 
                          lambda = const*2*sqrt(log(p)/length(ind2)))
Theta.init = Theta.re0$Theta.hat
Omega.tl2 = Trans.CLIME(X = Xt[ind2,], X.A, const = const,
                        X.til = Xt[1:(n0 - n00),], Theta.cl = Theta.init)
Ome_hat_trans_clime = (Omega.tl1 + Omega.tl2)/2

error.trans.clime = norm(Ome_hat_trans_clime - Ome_t,"F")^2/p
error.diag.trans.clime = sum(abs(diag(Ome_hat_trans_clime) - diag(Ome_t)))/p

##============================================
## Trans-Glasso
##============================================

# fit.trans.glasso = transglasso_unknownA(Xt,XS)
# Ome_hat_trans_glasso = fit.trans.glasso$Ome_hat_trans_glasso
# 
# error.trans.glasso = norm(Ome_hat_trans_glasso - Ome_t,"F")^2/p
# error.diag.trans.glasso = sum(abs(diag(Ome_hat_trans_glasso) - diag(Ome_t)))/p

##=====================================================
## loss
##=====================================================

error.precision.matrix = c(error.tar,error.NTL,
                           # error.trans.glasso,
                           error.trans.clime)
error.diag = c(error.diag.tar,error.diag.NTL,
               # error.diag.trans.glasso,
               error.diag.trans.clime)
names(error.precision.matrix) = names(error.diag) = 
  c("Target-GGM","NTL-GGM",
    # "Trans-Glasso",
    "Trans-CLIME")
error.precision.matrix
error.diag
