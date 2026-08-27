library(igraph)
library(ggraph)
library(ggplot2)
library(patchwork)
library(dplyr)

#----------------------------------------------------------
gap_range <- 1500:3000
gap_df <- subset(edge_info, edge_order %in% gap_range)

#========================================
# 1. 在 gap_df 中定义边类型
#========================================
gap_df$edge_type <- NA_character_

gap_df$edge_type[gap_df$`NTL-GGM_present` == 1 & gap_df$`Trans-CLIME_present` == 1] <- "shared"
gap_df$edge_type[gap_df$`NTL-GGM_present` == 0 & gap_df$`Trans-CLIME_present` == 1] <- "Trans-CLIME-only"
gap_df$edge_type[gap_df$`NTL-GGM_present` == 1 & gap_df$`Trans-CLIME_present` == 0] <- "NTL-GGM-only"
unique(gap_df$edge_type)
#========================================
# 2. 分三类边
#========================================
shared_edges  <- subset(gap_df, edge_type == "shared")
m2_only_edges <- subset(gap_df, edge_type == "Trans-CLIME-only")
m3_only_edges <- subset(gap_df, edge_type == "NTL-GGM-only")

#========================================
# 3. 为每类边定义排序分数
#========================================
if (nrow(shared_edges) > 0) {
  shared_edges$rank_score <- abs(shared_edges$`Trans-CLIME_value`) + abs(shared_edges$`NTL-GGM_value`)
}
if (nrow(m2_only_edges) > 0) {
  m2_only_edges$rank_score <- abs(m2_only_edges$`Trans-CLIME_value`)
}
if (nrow(m3_only_edges) > 0) {
  m3_only_edges$rank_score <- abs(m3_only_edges$`NTL-GGM_value`)
}

#========================================
# 4. 取 top 边
#   Trans-CLIME_only 作为重点，数量最多
#   NTL-GGM_only 保留少量
#========================================
select_top_by_quantile <- function(df, score_col, q = 0.9) {
  if (nrow(df) == 0) return(df)
  cutoff <- quantile(df[[score_col]], probs = q, na.rm = TRUE)
  df[df[[score_col]] >= cutoff, , drop = FALSE]
}

top_shared  <- select_top_by_quantile(shared_edges,  "rank_score", q = 0.6)
top_m2_only <- select_top_by_quantile(m2_only_edges, "rank_score", q = 0.98)
top_m3_only <- select_top_by_quantile(m3_only_edges, "rank_score", q = 0.95)

#========================================
# 5. 分别构造 Method-2 / Method-3 的作图边集
#========================================
plot_edges_m2 <- rbind(top_shared, top_m2_only)
plot_edges_m3 <- rbind(top_shared, top_m3_only)

# 节点用两张图的并集，保证可比
nodes_plot <- data.frame(
  name = sort(unique(c(
    plot_edges_m2$gene1, plot_edges_m2$gene2,
    plot_edges_m3$gene1, plot_edges_m3$gene2
  ))),
  stringsAsFactors = FALSE
)

#========================================
# 6. 构造两个图对象
#========================================
g_m2 <- graph_from_data_frame(
  d = data.frame(
    from = plot_edges_m2$gene1,
    to   = plot_edges_m2$gene2,
    edge_type = plot_edges_m2$edge_type,
    stringsAsFactors = FALSE
  ),
  vertices = nodes_plot,
  directed = FALSE
)

g_m3 <- graph_from_data_frame(
  d = data.frame(
    from = plot_edges_m3$gene1,
    to   = plot_edges_m3$gene2,
    edge_type = plot_edges_m3$edge_type,
    stringsAsFactors = FALSE
  ),
  vertices = nodes_plot,
  directed = FALSE
)

#========================================
# 7. 用并集图生成共同布局
#========================================
edges_union <- unique(rbind(
  data.frame(from = plot_edges_m2$gene1, to = plot_edges_m2$gene2),
  data.frame(from = plot_edges_m3$gene1, to = plot_edges_m3$gene2)
))

g_union <- graph_from_data_frame(
  d = edges_union,
  vertices = nodes_plot,
  directed = FALSE
)

set.seed(123)
lay_union <- layout_with_fr(g_union)

layout_df <- data.frame(
  name = V(g_union)$name,
  x = lay_union[, 1],
  y = lay_union[, 2],
  stringsAsFactors = FALSE
)

#========================================
# 8. 把共同布局赋给两张图
#========================================
V(g_m2)$x <- layout_df$x[match(V(g_m2)$name, layout_df$name)]
V(g_m2)$y <- layout_df$y[match(V(g_m2)$name, layout_df$name)]

V(g_m3)$x <- layout_df$x[match(V(g_m3)$name, layout_df$name)]
V(g_m3)$y <- layout_df$y[match(V(g_m3)$name, layout_df$name)]

lay_m2 <- create_layout(g_m2, layout = "manual", x = V(g_m2)$x, y = V(g_m2)$y)
lay_m3 <- create_layout(g_m3, layout = "manual", x = V(g_m3)$x, y = V(g_m3)$y)

# 所有节点都标注
lay_m2$label <- lay_m2$name
lay_m3$label <- lay_m3$name

#========================================
# 9. 画 Method-2 图
#========================================

#-------------------------------------------------------
library(ggforce)

region_red <- c("WNT3","SLIT3")
region_yellow <- c("FGF8","NDNF")
region_blue <- c("ARX","CSNK1D","WNT1")

lay_m2_region <- lay_m2 %>%
  mutate(region = case_when(
    name %in% region_red ~ "Region 1",
    name %in% region_yellow ~ "Region 2",
    name %in% region_blue ~ "Region 3",
    TRUE ~ NA_character_
  ))

p_m2 <- ggraph(lay_m2) +
  geom_edge_link(aes(color = edge_type), linewidth = 1, alpha = 0.9) +
  geom_node_point(size = 2.8, color = "#1A1A1A") +
  geom_node_text(aes(label = label), repel = TRUE, size = 3) +
  ggforce::geom_mark_ellipse(
    data = subset(lay_m2_region, !is.na(region)),
    aes(x = x, y = y, group = region),
    inherit.aes = FALSE,
    expand = unit(5, "mm"),
    con.cap = unit(0, "mm"),
    label.fontsize = 0,
    color = "#7A7A7A",
    linetype = "dashed",
    fill = NA,
    linewidth = 0.8
  ) +
  scale_edge_color_manual(
    values = c(shared = "red", `Trans-CLIME-only` = "black"),
    breaks = c("shared", "Trans-CLIME-only"),
    labels = c("Shared", "Trans-CLIME only")
  ) +
  theme_void() +
    labs( caption = "(a) Subgraph by Trans-CLIME", edge_color = "Edge type") +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.caption = element_text(face = "bold", size = 12, hjust = 0.5),
    plot.margin = margin(10, 30, 20, 10)
  ) 
  
  
lay_m3_region <- lay_m3 %>%
  mutate(region = case_when(
    name %in% region_red ~ "Region 1",
    name %in% region_yellow ~ "Region 2",
    name %in% region_blue ~ "Region 3",
    TRUE ~ NA_character_
  ))

p_m3 <- ggraph(lay_m3) +
  geom_edge_link(aes(color = edge_type), linewidth = 1, alpha = 0.9) +
  geom_node_point(size = 2.8, color = "#1A1A1A") +
  geom_node_text(aes(label = label), repel = TRUE, size = 3) +
  ggforce::geom_mark_ellipse(
    data = subset(lay_m3_region, !is.na(region)),
    aes(x = x, y = y, group = region),
    inherit.aes = FALSE,
    expand = unit(5, "mm"),
    con.cap = unit(0, "mm"),
    label.fontsize = 0,
    color = "#7A7A7A",
    linetype = "dashed",
    fill = NA,
    linewidth = 0.8
  ) +
  scale_edge_color_manual(
    values = c(shared = "red", `NTL-GGM-only` = "blue"),
    breaks = c("shared", "NTL-GGM-only"),
    labels = c("Shared", "NTL-GGM only")
  ) +
  theme_void() +
  labs( caption = "(b) Subgraph by NTL-GGM", edge_color = "Edge type") +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.caption = element_text(face = "bold", size = 12, hjust = 0.5),
    plot.margin = margin(10, 10, 20, 30)
  )

p_m2 + p_m3 + plot_layout(ncol = 2)


