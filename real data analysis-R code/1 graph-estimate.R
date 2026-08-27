## This file is used to visualize the edges detected by all methods in the first data split, 
## ordered by the score index.
#------------------------------------------
library(ggplot2)
#------------------------------------------

load("real-data-result.RData")

#---------------------------
# 1. 输入4个方法矩阵
#---------------------------
## The following four estimators are obtained from first data-splitting.

method_list <- list(
  "Target-GGM" = Ome_target_GGM,
  "Trans-CLIME" = Omega.trans.clime.hat,
  "NTL-GGM" = Ome_NTL_GGM_lasso,
  "Trans-Glasso-CV" = Ome_trans_glasso.hat
)

p = ncol(Ome_target_GGM)
sparsity.vec = 
  c((sum(Ome_target_GGM != 0) - p)/(p*(p - 1)),
  (sum(Ome_NTL_GGM_lasso != 0) - p)/(p*(p - 1)),
  (sum(Omega.trans.clime.hat != 0) - p)/(p*(p - 1)),
  (sum(Ome_trans_glasso.hat != 0) - p)/(p*(p - 1)))
names(sparsity.vec) = c("Target-GGM","NTL-GGM","Trans-CLIME","Trans-Glasso")

# 非零判定阈值
eps <- 1e-5
edge_counts <- sapply(method_list, function(mat) {
  idx <- upper.tri(mat, diag = FALSE)
  sum(abs(mat[idx]) > eps)
})

method_order <- names(sort(edge_counts, decreasing = FALSE))


#---------------------------
# 2. 提取每个矩阵上三角边
#---------------------------
get_upper_info <- function(mat, eps = 1e-8) {
  idx <- upper.tri(mat, diag = FALSE)
  
  val <- mat[idx]
  
  data.frame(
    i = row(mat)[idx],
    j = col(mat)[idx],
    value = val,
    present = as.integer(abs(val) > eps)
  )
}

edge_list <- lapply(method_list, get_upper_info, eps = eps)



#---------------------------
# 3. 计算每条边的总score
#---------------------------
n_edges <- nrow(edge_list[[1]])

score_df <- edge_list[[1]][, c("i", "j")]
score_df$score <- 0

for (mn in names(edge_list)) {
  score_df$score <- score_df$score + edge_list[[mn]]$present
}

#---------------------------
# 4. 次级排序：绝对值和
#---------------------------
score_df$abs_sum <- 0

for (mn in names(edge_list)) {
  score_df$abs_sum <- score_df$abs_sum + abs(edge_list[[mn]]$value)
}

# 先按score降序，再按abs_sum降序
ord <- order(score_df$score, score_df$abs_sum, decreasing = TRUE)

score_df <- score_df[ord, ]
score_df$edge_order <- seq_len(nrow(score_df))

#---------------------------
# 5. 所有方法按统一score顺序整理
#---------------------------
plot_df <- do.call(rbind, lapply(names(edge_list), function(mn) {
  df <- edge_list[[mn]][ord, ]
  df$edge_order <- seq_len(nrow(df))
  df$method <- mn
  df
}))

method_order <- c("Target-GGM", "NTL-GGM", "Trans-CLIME", "Trans-Glasso-CV")

plot_df$method <- factor(plot_df$method, levels = rev(method_order))

ggplot(plot_df[plot_df$present == 1, ],
       aes(x = edge_order, y = method, fill = method)) +
  geom_tile(width = 1, height = 0.85) +
  coord_cartesian(expand = FALSE) +
  labs(x = "Edge Index (ordered by score)", y = NULL) +
  theme_bw(base_size = 14) +
  theme(
    axis.text.y = element_text(face = "bold"),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.5),
    axis.line = element_line(color = "black"),
    panel.grid = element_blank(),
    legend.position = "none"
  )          
 
          
          
  
