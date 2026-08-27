

library(readxl)
# setwd("C:\\Users\\89261\\Nutstore\\1\\我的坚果云\\New folder\\NTL-real data")
df = read_xlsx("C:\\Users\\89261\\Nutstore\\1\\我的坚果云\\New folder\\NTL-real data\\GO-gene.xlsx",col_names = FALSE)
# 去掉第二列重复元素所在的行（保留第一次出现）
df = as.matrix(df)
df_unique <- df[!duplicated(df[,2]), ]

# 提取 intersect.gene 在第二列中对应的第一列名称
result <- df_unique[df_unique[,2] %in% intersect.gene, 1]

# 查看结果
gene.name = result

# =========================
# 0. 基本检查
# =========================
stopifnot(length(method_list) == 4)

p <- nrow(method_list[[1]])
stopifnot(all(sapply(method_list, nrow) == p))
stopifnot(all(sapply(method_list, ncol) == p))
stopifnot(length(gene.name) == p)

# 给矩阵加行列名
for (nm in names(method_list)) {
  rownames(method_list[[nm]]) <- gene.name
  colnames(method_list[[nm]]) <- gene.name
}


# =========================
# 1. 提取上三角边信息
# =========================
get_upper_edges <- function(mat, eps = 1e-8) {
  idx <- upper.tri(mat, diag = FALSE)
  
  data.frame(
    i = row(mat)[idx],
    j = col(mat)[idx],
    gene1 = rownames(mat)[row(mat)[idx]],
    gene2 = colnames(mat)[col(mat)[idx]],
    value = mat[idx],
    present = as.integer(abs(mat[idx]) > eps),
    stringsAsFactors = FALSE
  )
}

edge_list <- lapply(method_list, get_upper_edges, eps = eps)

# =========================
# 2. 建立统一边表
# =========================
edge_info <- edge_list[[1]][, c("i", "j", "gene1", "gene2")]

# 每条边的 score = 四个方法中的 present 之和
edge_info$score <- 0

# 次级排序指标：四个方法绝对值之和
edge_info$abs_sum <- 0

for (nm in names(edge_list)) {
  edge_info[[paste0(nm, "_present")]] <- edge_list[[nm]]$present
  edge_info[[paste0(nm, "_value")]]   <- edge_list[[nm]]$value
  
  edge_info$score   <- edge_info$score + edge_list[[nm]]$present
  edge_info$abs_sum <- edge_info$abs_sum + abs(edge_list[[nm]]$value)
}

# =========================
# 3. 排序
#    先按 score，再按 abs_sum
# =========================
ord <- order(edge_info$score, edge_info$abs_sum, decreasing = TRUE)

edge_info <- edge_info[ord, ]
edge_info$edge_order <- seq_len(nrow(edge_info))
# head(edge_info)



