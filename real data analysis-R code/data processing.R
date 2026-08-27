library(stringi)
library(glmnet)
library(ncvreg)
library(MVN)
#------------------------------------------------
load("E:\\brain-tissue.RData")

data.all$gene_id = stri_sub(data.all$gene_id,1,15)
ALL.tissue.name = data.all$tissue
ALL.tissue.name = ALL.tissue.name[!duplicated(ALL.tissue.name)]
brain.tissue.name = ALL.tissue.name

brain.tissue = list()
for (l in 1:length(brain.tissue.name)) {
  
  brain.tissue[[l]] = subset(data.all,tissue == brain.tissue.name[l])
  brain.tissue[[l]] = data.frame(t(na.omit(t(brain.tissue[[l]]))))
}

gene_interest = read.table("clipboard",header = F)

gene.name = list()
brain.tissue.test = list()
for (l in 1:length(brain.tissue.name)) {
  
  brain.tissue.test[[l]] = subset(brain.tissue[[l]],gene_id %in% gene_interest$V1)
  gene.name[[l]] = brain.tissue.test[[l]]$gene_id
}

intersect.gene = Reduce(intersect,gene.name);length(intersect.gene)
for (l in 1:length(brain.tissue.name)) {
  
  brain.tissue.test[[l]] = subset(brain.tissue.test[[l]],gene_id %in% intersect.gene)
  brain.tissue.test[[l]] = t(brain.tissue.test[[l]][,-c(1:5)])
  brain.tissue.test[[l]] = apply(brain.tissue.test[[l]],2,as.numeric)
}

p = ncol(brain.tissue.test[[1]])
p_result = rep(0,length(brain.tissue.test))
for (l in 1:length(brain.tissue.test)) {
  
  # for (j in 1:p) {
  
  #   fit.test = shapiro.test(brain.tissue.test[[l]][,j])
  #   p_result[j,l] = fit.test$p.value
  # }
  # l = 4
  data = as.data.frame(brain.tissue.test[[l]])
  # data = rmvnorm(120,mean = rep(0,20),diag(20))
  fit = mvn(data, mvn_test = "hz")
  value = fit$multivariate_normality[3]
  p_result[l] = as.numeric(value[[1]])
}
# p_result

id = which(p_result > 0.1);
# id;ALL.tissue.name[id];p
use.tissue = brain.tissue.test[id]
n.vec = c()
for (l in seq_along(use.tissue)) {
  
  n.vec[l] = nrow(use.tissue[[l]])
  use.tissue[[l]] = scale(use.tissue[[l]])
}

delete.id = which(n.vec >= 100)
use.tissue = use.tissue[delete.id]

