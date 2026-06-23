############################################################
# FUNCTIONS ONLY
# CASE-BASED FRT + RULE-BASED TEST APPLICATION
############################################################

############################
# 0. PACKAGES
############################
required_packages <- c(
  "rpart",
  "openxlsx",
  "randomForest",
  "xgboost",
  "Cubist",
  "BART"
)

for (p in required_packages) {
  if (!require(p, character.only = TRUE)) {
    install.packages(p, dependencies = TRUE)
    library(p, character.only = TRUE)
  }
}

############################
# 1. GFN ARITHMETIC FUNCTIONS
############################
GFN.add <- function(A, B) c(A[1] + B[1], A[2] + B[2])

GFN.sub <- function(A, B) c(A[1] - B[1], A[2] + B[2])

GFN.multi <- function(A, B) {
  c(
    A[1] * B[1],
    (B[2] * A[1]^2) + (A[2] * B[1]^2) + (B[2] * A[2])
  )
}

GFN.div <- function(A, B) {
  mean_val <- A[1] * ((1 / B[1]) + (B[2] / B[1]^3))
  variance_val <- (A[1]^2 * (1 / B[1]^4) * B[2]) +
    ((1 / B[1]^2) * A[2]) -
    (A[2] * (1 / B[1]^4) * B[2])
  c(mean_val, variance_val)
}

############################
# 2. FUZZIFICATION
############################
fuzzify <- function(X, variance) {
  X <- as.numeric(X)
  matrix(
    c(X, rep(variance, length(X))),
    ncol = 2,
    byrow = FALSE,
    dimnames = list(NULL, c("Mean", "Variance"))
  )
}

defuzzify <- function(gfn, k, m, symmetry.threshold) {
  mean_val <- gfn[1]
  var_val <- gfn[2]
  
  if (is.na(mean_val) || is.na(var_val) || var_val <= 0) {
    return(mean_val)
  }
  
  delta_val <- abs(mean_val / sqrt(var_val))
  
  if (delta_val > symmetry.threshold) {
    return(mean_val)
  }
  
  adj_factor <- m / (1 + exp(-k * (delta_val - symmetry.threshold)))
  mean_val + adj_factor * var_val
}

############################
# 3. FUZZY ERROR FUNCTIONS
############################
GFN.abs <- function(gfn) c(abs(gfn[1]), gfn[2])

GFN.sqrt <- function(gfn) c(sqrt(gfn[1]), gfn[2] / (2 * sqrt(gfn[1]) + 1e-6))

defuzzy_errors <- function(gfn, k = 2, m = 1, symmetry.threshold = 1) {
  defuzzify(gfn, k, m, symmetry.threshold)
}

MSEe_GFN <- function(Y_true, Y_pred) {
  n <- nrow(Y_true)
  mse <- c(0, 0)
  
  for (i in seq_len(n)) {
    diff <- GFN.sub(Y_pred[i, ], Y_true[i, ])
    mse <- GFN.add(mse, GFN.multi(diff, diff))
  }
  
  GFN.div(mse, c(n, 0))
}

MAEe_GFN <- function(Y_true, Y_pred) {
  n <- nrow(Y_true)
  mae <- c(0, 0)
  
  for (i in seq_len(n)) {
    diff <- GFN.sub(Y_pred[i, ], Y_true[i, ])
    mae <- GFN.add(mae, GFN.abs(diff))
  }
  
  GFN.div(mae, c(n, 0))
}

MAPEe_GFN <- function(Y_true, Y_pred) {
  n <- nrow(Y_true)
  mape <- c(0, 0)
  
  for (i in seq_len(n)) {
    diff <- GFN.sub(Y_pred[i, ], Y_true[i, ])
    abs_diff <- GFN.abs(diff)
    denom <- GFN.add(Y_true[i, ], c(1e-6, 0))
    ratio <- GFN.div(abs_diff, denom)
    mape <- GFN.add(mape, ratio)
  }
  
  result <- GFN.div(mape, c(n, 0))
  GFN.multi(result, c(100, 0))
}

FMSE_score <- function(Y_true, Y_pred, k = 2, m = 1, symmetry.threshold = 1) {
  defuzzy_errors(MSEe_GFN(Y_true, Y_pred), k, m, symmetry.threshold)
}

FRMSE_score <- function(Y_true, Y_pred, k = 2, m = 1, symmetry.threshold = 1) {
  fmse <- MSEe_GFN(Y_true, Y_pred)
  defuzzy_errors(GFN.sqrt(fmse), k, m, symmetry.threshold)
}

FMAE_score <- function(Y_true, Y_pred, k = 2, m = 1, symmetry.threshold = 1) {
  defuzzy_errors(MAEe_GFN(Y_true, Y_pred), k, m, symmetry.threshold)
}

FMAPE_score <- function(Y_true, Y_pred, k = 2, m = 1, symmetry.threshold = 1) {
  defuzzy_errors(MAPEe_GFN(Y_true, Y_pred), k, m, symmetry.threshold)
}

calc_all_fuzzy_errors <- function(Y_true, Y_pred, k = 2, m = 1, symmetry.threshold = 1) {
  fmse <- MSEe_GFN(Y_true, Y_pred)
  fmae <- MAEe_GFN(Y_true, Y_pred)
  fmape <- MAPEe_GFN(Y_true, Y_pred)
  
  list(
    FMSE = defuzzy_errors(fmse, k, m, symmetry.threshold),
    FRMSE = defuzzy_errors(GFN.sqrt(fmse), k, m, symmetry.threshold),
    FMAE = defuzzy_errors(fmae, k, m, symmetry.threshold),
    FMAPE = defuzzy_errors(fmape, k, m, symmetry.threshold)
  )
}

############################
# 4. CRISP ERROR FUNCTIONS
############################
calc_crisp_errors <- function(y_true, y_pred) {
  mse <- mean((y_true - y_pred)^2, na.rm = TRUE)
  rmse <- sqrt(mse)
  mae <- mean(abs(y_true - y_pred), na.rm = TRUE)
  mape <- mean(abs((y_true - y_pred) / (y_true + 1e-6)), na.rm = TRUE) * 100
  
  list(
    MSE = mse,
    RMSE = rmse,
    MAE = mae,
    MAPE = mape
  )
}

############################
# 5. GFN DISTANCE
############################
GFN.KL <- function(A, B) {
  mu1 <- A[1]
  v1 <- A[2]
  mu2 <- B[1]
  v2 <- B[2]
  0.5 * (log(v2 / v1) + (v1 + (mu1 - mu2)^2) / v2 - 1)
}

############################
# 6. GFN MEAN
############################
GFN.mean_of_rows <- function(GFN_mat) {
  n <- nrow(GFN_mat)
  mu <- mean(GFN_mat[, 1])
  var <- sum(GFN_mat[, 2]) / (n^2)
  c(mu, var)
}

############################
# 7. BUILD CRISP TREE
############################
build_crisp_tree <- function(df, y_col,
                             control = rpart.control(
                               cp = 0.001,
                               minsplit = 8,
                               minbucket = 4,
                               maxdepth = 10,
                               xval = 0
                             )) {
  rpart(
    as.formula(paste(y_col, "~ .")),
    data = df,
    method = "anova",
    control = control
  )
}

############################
# 8. INTERNAL NODES
############################
get_internal_nodes <- function(tree) {
  as.numeric(rownames(tree$frame)[tree$frame$var != "<leaf>"])
}

############################
# 9. SPLIT INFORMATION
############################
get_split_info <- function(tree, node) {
  fr <- tree$frame
  frame_rows <- rownames(fr)
  node_row <- match(as.character(node), frame_rows)
  
  if (is.na(node_row)) {
    stop("Node not found in tree frame: ", node)
  }
  
  if (fr$var[node_row] == "<leaf>") {
    stop("Requested split info for a leaf node: ", node)
  }
  
  internal_row_idx <- which(fr$var != "<leaf>")
  internal_pos <- match(node_row, internal_row_idx)
  
  if (is.na(internal_pos)) {
    stop("Internal node position not found for node: ", node)
  }
  
  # rpart stores primary, competitor, and surrogate splits consecutively.
  # We need the row offset of the primary split for this specific node,
  # not just the ordinal index of the node among internal nodes.
  if (internal_pos == 1) {
    row_in_splits <- 1
  } else {
    prev_internal_rows <- internal_row_idx[seq_len(internal_pos - 1)]
    row_in_splits <- 1 + sum(
      fr$ncompete[prev_internal_rows] + fr$nsurrogate[prev_internal_rows] + 1
    )
  }
  
  list(
    var = as.character(fr[as.character(node), "var"]),
    s = as.numeric(tree$splits[row_in_splits, "index"]),
    ncat = as.numeric(tree$splits[row_in_splits, "ncat"])
  )
}

############################
# 10. ROUTING WITH EXPLICIT FLIPS
############################
route_one <- function(tree, x_row, obs_id, flips = list()) {
  fr <- tree$frame
  node <- 1
  decisions <- list()
  
  while (fr[as.character(node), "var"] != "<leaf>") {
    info <- get_split_info(tree, node)
    x_val <- as.numeric(x_row[[info$var]])
    less_flag <- x_val < info$s
    go_left <- if (info$ncat < 0) less_flag else !less_flag
    
    if (!is.null(flips[[as.character(node)]]) &&
        obs_id %in% flips[[as.character(node)]]) {
      go_left <- !go_left
    }
    
    decisions[[as.character(node)]] <- if (go_left) "L" else "R"
    node <- if (go_left) node * 2 else node * 2 + 1
  }
  
  list(leaf = node, decisions = decisions)
}

route_all <- function(tree, X_df, flips = list()) {
  X_df <- as.data.frame(X_df)
  n <- nrow(X_df)
  internal_nodes <- get_internal_nodes(tree)
  
  decisions_by_node <- setNames(
    lapply(as.character(internal_nodes), function(k) rep(NA, n)),
    as.character(internal_nodes)
  )
  
  leaf_ids <- integer(n)
  
  for (i in seq_len(n)) {
    res <- route_one(tree, X_df[i, , drop = FALSE], i, flips)
    leaf_ids[i] <- res$leaf
    
    for (k in names(res$decisions)) {
      decisions_by_node[[k]][i] <- res$decisions[[k]]
    }
  }
  
  list(
    leaf = leaf_ids,
    decisions_by_node = decisions_by_node
  )
}

############################
# 11. PROTOTYPES
############################
compute_leaf_prototypes <- function(leaf_ids, GFN_Y) {
  prot <- list()
  
  for (leaf in sort(unique(leaf_ids))) {
    prot[[as.character(leaf)]] <- GFN.mean_of_rows(
      GFN_Y[leaf_ids == leaf, , drop = FALSE]
    )
  }
  
  prot
}

compute_node_prototypes <- function(leaf_ids, GFN_Y) {
  node_members <- list()
  
  for (i in seq_along(leaf_ids)) {
    node <- leaf_ids[i]
    
    repeat {
      key <- as.character(node)
      
      if (is.null(node_members[[key]])) {
        node_members[[key]] <- i
      } else {
        node_members[[key]] <- c(node_members[[key]], i)
      }
      
      if (node == 1) break
      node <- floor(node / 2)
    }
  }
  
  node_prototypes <- list()
  
  for (key in names(node_members)) {
    idx <- unique(node_members[[key]])
    node_prototypes[[key]] <- GFN.mean_of_rows(GFN_Y[idx, , drop = FALSE])
  }
  
  node_prototypes
}

############################
# 12. PREDICT FROM LEAVES
############################
predict_gfn_from_leaves <- function(leaf_ids, leaf_prototypes, node_prototypes, global_proto) {
  pred_list <- lapply(leaf_ids, function(l) {
    key <- as.character(l)
    
    if (!is.null(leaf_prototypes[[key]])) {
      return(leaf_prototypes[[key]])
    }
    
    parent <- floor(l / 2)
    
    while (parent >= 1) {
      parent_key <- as.character(parent)
      
      if (!is.null(node_prototypes[[parent_key]])) {
        return(node_prototypes[[parent_key]])
      }
      
      parent <- floor(parent / 2)
    }
    
    global_proto
  })
  
  do.call(rbind, pred_list)
}

predict_tree_gfn <- function(tree, X_df, GFN_Y_ref, flips = list()) {
  routed <- route_all(tree, X_df, flips)
  leaf_proto <- compute_leaf_prototypes(routed$leaf, GFN_Y_ref)
  node_proto <- compute_node_prototypes(routed$leaf, GFN_Y_ref)
  global_proto <- GFN.mean_of_rows(GFN_Y_ref)
  
  pred <- predict_gfn_from_leaves(
    routed$leaf,
    leaf_proto,
    node_proto,
    global_proto
  )
  
  list(
    pred = pred,
    routed = routed,
    leaf_proto = leaf_proto,
    node_proto = node_proto,
    global_proto = global_proto
  )
}

############################
# 13. NEAR SPLIT
############################
near_split_indices <- function(GFN_Xj, split_gfn, eps_f) {
  which(
    apply(GFN_Xj, 1, function(x) {
      GFN.KL(x, split_gfn) <= eps_f
    })
  )
}

is_near_split_single <- function(gfn_x, split_gfn, eps_f) {
  GFN.KL(gfn_x, split_gfn) <= eps_f
}

############################
# 14. SUPPORT / CONFIDENCE
############################
compute_transition_rules <- function(AssignedSide, TargetConsistentSide) {
  m <- length(AssignedSide)
  
  if (m == 0) {
    return(list(
      supp_R_to_L = 0,
      conf_R_to_L = 0,
      supp_L_to_R = 0,
      conf_L_to_R = 0
    ))
  }
  
  list(
    supp_R_to_L = sum(AssignedSide == "R" & TargetConsistentSide == "L") / m,
    conf_R_to_L = ifelse(
      sum(AssignedSide == "R") == 0,
      0,
      sum(AssignedSide == "R" & TargetConsistentSide == "L") / sum(AssignedSide == "R")
    ),
    supp_L_to_R = sum(AssignedSide == "L" & TargetConsistentSide == "R") / m,
    conf_L_to_R = ifelse(
      sum(AssignedSide == "L") == 0,
      0,
      sum(AssignedSide == "L" & TargetConsistentSide == "R") / sum(AssignedSide == "L")
    )
  )
}

############################
# 14B. VALIDATION / RULE HELPERS
############################
subset_gfn_list <- function(GFN_X_list, idx) {
  out <- lapply(GFN_X_list, function(mat) mat[idx, , drop = FALSE])
  out
}

make_validation_split <- function(n, validation_fraction = 0.20, seed = 123) {
  set.seed(seed)
  
  n_val <- max(1, floor(n * validation_fraction))
  val_idx <- sort(sample(seq_len(n), size = n_val, replace = FALSE))
  train_idx <- setdiff(seq_len(n), val_idx)
  
  list(train_idx = train_idx, val_idx = val_idx)
}

build_flip_rules_from_cases <- function(
    accepted_cases,
    eps_f,
    min_rule_cases = 3,
    max_rule_span = Inf) {
  
  flip_rules <- list()
  
  for (node_key in names(accepted_cases)) {
    df_node <- accepted_cases[[node_key]]
    if (is.null(df_node) || nrow(df_node) == 0) next
    
    node_rule <- list(
      var = unique(df_node$var)[1],
      split_value = unique(df_node$split_value)[1],
      eps_f = eps_f
    )
    
    df_r2l <- df_node[df_node$direction == "R_to_L", , drop = FALSE]
    df_l2r <- df_node[df_node$direction == "L_to_R", , drop = FALSE]
    
    if (nrow(df_r2l) >= min_rule_cases) {
      span_r2l <- max(df_r2l$x_value, na.rm = TRUE) - min(df_r2l$x_value, na.rm = TRUE)
      
      if (span_r2l <= max_rule_span) {
        node_rule$R_to_L <- list(
          active = TRUE,
          x_min = min(df_r2l$x_value, na.rm = TRUE),
          x_max = max(df_r2l$x_value, na.rm = TRUE),
          n_cases = nrow(df_r2l)
        )
      }
    }
    
    if (nrow(df_l2r) >= min_rule_cases) {
      span_l2r <- max(df_l2r$x_value, na.rm = TRUE) - min(df_l2r$x_value, na.rm = TRUE)
      
      if (span_l2r <= max_rule_span) {
        node_rule$L_to_R <- list(
          active = TRUE,
          x_min = min(df_l2r$x_value, na.rm = TRUE),
          x_max = max(df_l2r$x_value, na.rm = TRUE),
          n_cases = nrow(df_l2r)
        )
      }
    }
    
    if (!is.null(node_rule$R_to_L) || !is.null(node_rule$L_to_R)) {
      flip_rules[[node_key]] <- node_rule
    }
  }
  
  flip_rules
}

evaluate_rule_based_configuration <- function(
    tree,
    X_rule,
    GFN_X_rule,
    GFN_Y_rule,
    X_eval,
    GFN_X_eval,
    GFN_Y_eval,
    y_eval_crisp,
    accepted_cases,
    eps_f,
    accept_metric = c("MSE", "FMSE"),
    defuzz_k = 5,
    defuzz_m = 1,
    defuzz_threshold = 2,
    min_rule_cases = 3,
    max_rule_span = Inf,
    flip_penalty = 0.001,
    flips_count = 0) {
  
  accept_metric <- match.arg(accept_metric)
  
  flip_rules <- build_flip_rules_from_cases(
    accepted_cases = accepted_cases,
    eps_f = eps_f,
    min_rule_cases = min_rule_cases,
    max_rule_span = max_rule_span
  )
  
  routed_rule <- route_all_with_rules(
    tree = tree,
    X_df = X_rule,
    GFN_X_list = GFN_X_rule,
    flip_rules = flip_rules
  )
  
  leaf_proto_rule <- compute_leaf_prototypes(routed_rule$leaf, GFN_Y_rule)
  node_proto_rule <- compute_node_prototypes(routed_rule$leaf, GFN_Y_rule)
  global_proto_rule <- GFN.mean_of_rows(GFN_Y_rule)
  
  routed_eval <- route_all_with_rules(
    tree = tree,
    X_df = X_eval,
    GFN_X_list = GFN_X_eval,
    flip_rules = flip_rules
  )
  
  Y_pred_eval <- predict_gfn_from_leaves(
    routed_eval$leaf,
    leaf_proto_rule,
    node_proto_rule,
    global_proto_rule
  )
  
  if (accept_metric == "FMSE") {
    score <- FMSE_score(
      GFN_Y_eval,
      Y_pred_eval,
      k = defuzz_k,
      m = defuzz_m,
      symmetry.threshold = defuzz_threshold
    )
  } else {
    crisp_pred <- apply(Y_pred_eval, 1, function(gfn) {
      defuzzify(gfn, k = defuzz_k, m = defuzz_m, symmetry.threshold = defuzz_threshold)
    })
    score <- calc_crisp_errors(y_eval_crisp, crisp_pred)$MSE
  }
  
  penalized_score <- score + flip_penalty * flips_count
  
  list(
    score = penalized_score,
    raw_score = score,
    flip_rules = flip_rules,
    Y_pred_eval = Y_pred_eval
  )
}

############################
# 15. CASE-BASED TRAIN SCORING
############################
score_case_based_tree <- function(
    tree,
    X_df,
    GFN_Y,
    y_true_crisp,
    flips = list(),
    accept_metric = c("MSE", "FMSE"),
    defuzz_k = 5,
    defuzz_m = 1,
    defuzz_threshold = 2) {
  
  accept_metric <- match.arg(accept_metric)
  pred_obj <- predict_tree_gfn(tree, X_df, GFN_Y, flips)
  
  if (accept_metric == "FMSE") {
    score <- FMSE_score(
      GFN_Y,
      pred_obj$pred,
      k = defuzz_k,
      m = defuzz_m,
      symmetry.threshold = defuzz_threshold
    )
  } else {
    crisp_pred <- apply(pred_obj$pred, 1, function(gfn) {
      defuzzify(gfn, k = defuzz_k, m = defuzz_m, symmetry.threshold = defuzz_threshold)
    })
    score <- calc_crisp_errors(y_true_crisp, crisp_pred)$MSE
  }
  
  list(score = score, pred_obj = pred_obj)
}

############################
# 16. CASE-BASED TRAIN UPDATE
############################
update_tree_assignments_casewise <- function(
    tree,
    X_df,
    GFN_X_list,
    GFN_Y,
    y_true_crisp,
    eps_f = 0.35,
    tau_supp = 0.03,
    tau_conf = 0.06,
    max_iter = 10,
    accept_metric = c("MSE", "FMSE"),
    defuzz_k = 5,
    defuzz_m = 1,
    defuzz_threshold = 2,
    validation_fraction = 0.20,
    validation_seed = 123,
    min_rule_cases = 3,
    max_rule_span = Inf,
    flip_penalty = 0.001,
    verbose = TRUE) {
  
  accept_metric <- match.arg(accept_metric)
  X_df <- as.data.frame(X_df)
  GFN_Y <- as.matrix(GFN_Y)
  
  split_obj <- make_validation_split(
    n = nrow(X_df),
    validation_fraction = validation_fraction,
    seed = validation_seed
  )
  
  train_idx <- split_obj$train_idx
  val_idx <- split_obj$val_idx
  
  X_rule <- X_df[train_idx, , drop = FALSE]
  X_val <- X_df[val_idx, , drop = FALSE]
  
  GFN_X_rule <- subset_gfn_list(GFN_X_list, train_idx)
  GFN_X_val <- subset_gfn_list(GFN_X_list, val_idx)
  
  GFN_Y_rule <- GFN_Y[train_idx, , drop = FALSE]
  GFN_Y_val <- GFN_Y[val_idx, , drop = FALSE]
  
  y_rule_crisp <- y_true_crisp[train_idx]
  y_val_crisp <- y_true_crisp[val_idx]
  
  flips <- list()
  accepted_cases <- list()
  score_history <- numeric(0)
  internal_nodes <- get_internal_nodes(tree)
  
  current_eval <- evaluate_rule_based_configuration(
    tree = tree,
    X_rule = X_rule,
    GFN_X_rule = GFN_X_rule,
    GFN_Y_rule = GFN_Y_rule,
    X_eval = X_val,
    GFN_X_eval = GFN_X_val,
    GFN_Y_eval = GFN_Y_val,
    y_eval_crisp = y_val_crisp,
    accepted_cases = accepted_cases,
    eps_f = eps_f,
    accept_metric = accept_metric,
    defuzz_k = defuzz_k,
    defuzz_m = defuzz_m,
    defuzz_threshold = defuzz_threshold,
    min_rule_cases = min_rule_cases,
    max_rule_span = max_rule_span,
    flip_penalty = flip_penalty,
    flips_count = 0
  )
  
  best_score <- current_eval$score
  score_history <- c(score_history, best_score)
  
  for (it in seq_len(max_iter)) {
    changed_total <- 0
    routed <- route_all(tree, X_rule, flips)
    
    for (node in internal_nodes) {
      info <- get_split_info(tree, node)
      var <- info$var
      s_crisp <- info$s
      split_gfn <- c(s_crisp, GFN_X_rule[[var]][1, 2])
      node_key <- as.character(node)
      node_decisions <- routed$decisions_by_node[[node_key]]
      
      Nv <- intersect(
        which(!is.na(node_decisions)),
        near_split_indices(GFN_X_rule[[var]], split_gfn, eps_f)
      )
      
      if (!length(Nv)) next
      
      AssignedSide <- node_decisions[Nv]
      
      left_idx <- which(node_decisions == "L")
      right_idx <- which(node_decisions == "R")
      
      if (!length(left_idx) || !length(right_idx)) next
      
      muL <- GFN.mean_of_rows(GFN_Y_rule[left_idx, , drop = FALSE])
      muR <- GFN.mean_of_rows(GFN_Y_rule[right_idx, , drop = FALSE])
      
      TargetConsistentSide <- ifelse(
        sapply(Nv, function(i) GFN.KL(GFN_Y_rule[i, ], muL) <= GFN.KL(GFN_Y_rule[i, ], muR)),
        "L",
        "R"
      )
      
      rules <- compute_transition_rules(AssignedSide, TargetConsistentSide)
      case_order <- integer(0)
      case_dir <- character(0)
      
      if (rules$supp_R_to_L >= tau_supp && rules$conf_R_to_L >= tau_conf) {
        idx <- Nv[AssignedSide == "R" & TargetConsistentSide == "L"]
        case_order <- c(case_order, idx)
        case_dir <- c(case_dir, rep("R_to_L", length(idx)))
      }
      
      if (rules$supp_L_to_R >= tau_supp && rules$conf_L_to_R >= tau_conf) {
        idx <- Nv[AssignedSide == "L" & TargetConsistentSide == "R"]
        case_order <- c(case_order, idx)
        case_dir <- c(case_dir, rep("L_to_R", length(idx)))
      }
      
      if (!length(case_order)) next
      
      for (j in seq_along(case_order)) {
        obs <- case_order[j]
        obs_dir <- case_dir[j]
        
        flips_trial <- flips
        flips_trial[[node_key]] <- sort(unique(c(flips[[node_key]], obs)))
        
        accepted_cases_trial <- accepted_cases
        
        new_row <- data.frame(
          obs_id = obs,
          node = node,
          var = var,
          split_value = s_crisp,
          x_value = as.numeric(X_rule[obs, var]),
          direction = obs_dir,
          stringsAsFactors = FALSE
        )
        
        if (is.null(accepted_cases_trial[[node_key]])) {
          accepted_cases_trial[[node_key]] <- new_row
        } else {
          accepted_cases_trial[[node_key]] <- rbind(accepted_cases_trial[[node_key]], new_row)
        }
        
        trial_eval <- evaluate_rule_based_configuration(
          tree = tree,
          X_rule = X_rule,
          GFN_X_rule = GFN_X_rule,
          GFN_Y_rule = GFN_Y_rule,
          X_eval = X_val,
          GFN_X_eval = GFN_X_val,
          GFN_Y_eval = GFN_Y_val,
          y_eval_crisp = y_val_crisp,
          accepted_cases = accepted_cases_trial,
          eps_f = eps_f,
          accept_metric = accept_metric,
          defuzz_k = defuzz_k,
          defuzz_m = defuzz_m,
          defuzz_threshold = defuzz_threshold,
          min_rule_cases = min_rule_cases,
          max_rule_span = max_rule_span,
          flip_penalty = flip_penalty,
          flips_count = length(unlist(flips_trial))
        )
        
        if (trial_eval$score < best_score) {
          flips <- flips_trial
          accepted_cases <- accepted_cases_trial
          best_score <- trial_eval$score
          changed_total <- changed_total + 1
          routed <- route_all(tree, X_rule, flips)
        }
      }
    }
    
    score_history <- c(score_history, best_score)
    
    if (verbose) {
      cat(
        "Iteration:", it,
        "Acceptance metric:", accept_metric,
        "Validation penalized score:", best_score,
        "Accepted flips:", changed_total,
        "\n"
      )
    }
    
    if (!changed_total) break
  }
  
  final_flip_rules <- build_flip_rules_from_cases(
    accepted_cases = accepted_cases,
    eps_f = eps_f,
    min_rule_cases = min_rule_cases,
    max_rule_span = max_rule_span
  )
  
  list(
    tree = tree,
    flips = flips,
    flip_rules = final_flip_rules,
    score_history = score_history,
    accept_metric = accept_metric,
    accepted_cases = accepted_cases,
    train_idx = train_idx,
    val_idx = val_idx
  )
}


############################
# 17. ROUTING WITH LEARNED X-ONLY RULES
############################
route_one_with_rules <- function(
    tree,
    x_row,
    obs_id,
    GFN_X_list,
    flip_rules) {
  
  fr <- tree$frame
  node <- 1
  decisions <- list()
  
  while (fr[as.character(node), "var"] != "<leaf>") {
    info <- get_split_info(tree, node)
    node_key <- as.character(node)
    var <- info$var
    s_crisp <- info$s
    
    x_val <- as.numeric(x_row[[var]])
    less_flag <- x_val < s_crisp
    AssignedSide <- if (info$ncat < 0) {
      if (less_flag) "L" else "R"
    } else {
      if (less_flag) "R" else "L"
    }
    final_side <- AssignedSide
    
    rule <- flip_rules[[node_key]]
    
    if (!is.null(rule)) {
      split_gfn <- c(s_crisp, GFN_X_list[[var]][obs_id, 2])
      near_flag <- is_near_split_single(
        GFN_X_list[[var]][obs_id, ],
        split_gfn,
        rule$eps_f
      )
      
      if (near_flag) {
        if (
          AssignedSide == "R" &&
          !is.null(rule$R_to_L) &&
          isTRUE(rule$R_to_L$active) &&
          x_val >= rule$R_to_L$x_min &&
          x_val <= rule$R_to_L$x_max
        ) {
          final_side <- "L"
        }
        
        if (
          AssignedSide == "L" &&
          !is.null(rule$L_to_R) &&
          isTRUE(rule$L_to_R$active) &&
          x_val >= rule$L_to_R$x_min &&
          x_val <= rule$L_to_R$x_max
        ) {
          final_side <- "R"
        }
      }
    }
    
    decisions[[node_key]] <- final_side
    node <- if (final_side == "L") node * 2 else node * 2 + 1
  }
  
  list(leaf = node, decisions = decisions)
}

route_all_with_rules <- function(tree, X_df, GFN_X_list, flip_rules) {
  X_df <- as.data.frame(X_df)
  n <- nrow(X_df)
  internal_nodes <- get_internal_nodes(tree)
  
  decisions_by_node <- setNames(
    lapply(as.character(internal_nodes), function(k) rep(NA, n)),
    as.character(internal_nodes)
  )
  
  leaf_ids <- integer(n)
  
  for (i in seq_len(n)) {
    res <- route_one_with_rules(
      tree = tree,
      x_row = X_df[i, , drop = FALSE],
      obs_id = i,
      GFN_X_list = GFN_X_list,
      flip_rules = flip_rules
    )
    
    leaf_ids[i] <- res$leaf
    
    for (k in names(res$decisions)) {
      decisions_by_node[[k]][i] <- res$decisions[[k]]
    }
  }
  
  list(
    leaf = leaf_ids,
    decisions_by_node = decisions_by_node
  )
}

############################
# 18. PREPROCESSING HELPERS
############################
find_target_name <- function(df) {
  candidates <- c("Target", "Y", "y")
  found <- intersect(candidates, names(df))
  if (length(found) == 0) return(NA_character_)
  found[1]
}

remove_id_columns <- function(df) {
  keep <- !grepl("^(id|ID|Id|iD|index|Index|INDEX)$", names(df))
  df[, keep, drop = FALSE]
}

encode_train_test_numeric <- function(train_df, test_df, target_name_train, target_name_test) {
  names(train_df)[names(train_df) == target_name_train] <- ".target"
  names(test_df)[names(test_df) == target_name_test] <- ".target"
  
  predictor_names <- union(
    setdiff(names(train_df), ".target"),
    setdiff(names(test_df), ".target")
  )
  
  for (nm in setdiff(predictor_names, names(train_df))) train_df[[nm]] <- NA
  for (nm in setdiff(predictor_names, names(test_df))) test_df[[nm]] <- NA
  
  train_df <- train_df[, c(predictor_names, ".target"), drop = FALSE]
  test_df <- test_df[, c(predictor_names, ".target"), drop = FALSE]
  
  for (nm in predictor_names) {
    tr_col <- train_df[[nm]]
    te_col <- test_df[[nm]]
    
    if (is.character(tr_col) || is.factor(tr_col) ||
        is.character(te_col) || is.factor(te_col)) {
      combined_levels <- unique(c(as.character(tr_col), as.character(te_col)))
      combined_levels <- combined_levels[!is.na(combined_levels)]
      
      train_df[[nm]] <- as.numeric(factor(as.character(tr_col), levels = combined_levels))
      test_df[[nm]] <- as.numeric(factor(as.character(te_col), levels = combined_levels))
    } else {
      train_df[[nm]] <- suppressWarnings(as.numeric(tr_col))
      test_df[[nm]] <- suppressWarnings(as.numeric(te_col))
    }
  }
  
  if (is.character(train_df$.target) || is.factor(train_df$.target) ||
      is.character(test_df$.target) || is.factor(test_df$.target)) {
    target_levels <- unique(c(as.character(train_df$.target), as.character(test_df$.target)))
    target_levels <- target_levels[!is.na(target_levels)]
    
    train_df$.target <- as.numeric(factor(as.character(train_df$.target), levels = target_levels))
    test_df$.target <- as.numeric(factor(as.character(test_df$.target), levels = target_levels))
  } else {
    train_df$.target <- suppressWarnings(as.numeric(train_df$.target))
    test_df$.target <- suppressWarnings(as.numeric(test_df$.target))
  }
  
  list(train = train_df, test = test_df)
}

impute_by_train_median <- function(X_train, X_test) {
  for (nm in names(X_train)) {
    med <- median(X_train[[nm]], na.rm = TRUE)
    if (!is.finite(med)) med <- 0
    
    X_train[[nm]][is.na(X_train[[nm]])] <- med
    X_test[[nm]][is.na(X_test[[nm]])] <- med
  }
  
  list(X_train = X_train, X_test = X_test)
}

scale_train_test <- function(X_train, X_test, Y_train, Y_test) {
  X_train_mat <- scale(X_train)
  x_center <- attr(X_train_mat, "scaled:center")
  x_scale <- attr(X_train_mat, "scaled:scale")
  x_scale[x_scale == 0 | is.na(x_scale)] <- 1
  
  X_train_scaled <- scale(X_train, center = x_center, scale = x_scale)
  X_test_scaled <- scale(X_test, center = x_center, scale = x_scale)
  
  y_center <- mean(Y_train, na.rm = TRUE)
  y_scale <- sd(Y_train, na.rm = TRUE)
  if (!is.finite(y_scale) || y_scale == 0) y_scale <- 1
  
  Y_train_scaled <- (Y_train - y_center) / y_scale
  Y_test_scaled <- (Y_test - y_center) / y_scale
  
  list(
    X_train = as.data.frame(X_train_scaled),
    X_test = as.data.frame(X_test_scaled),
    Y_train = as.numeric(Y_train_scaled),
    Y_test = as.numeric(Y_test_scaled),
    x_center = x_center,
    x_scale = x_scale,
    y_center = y_center,
    y_scale = y_scale
  )
}

prepare_train_test_data <- function(df_train, df_test) {
  df_train <- remove_id_columns(df_train)
  df_test <- remove_id_columns(df_test)
  
  target_train <- find_target_name(df_train)
  target_test <- find_target_name(df_test)
  
  if (is.na(target_train) || is.na(target_test)) {
    cat("Target bulunamadi.\n")
    return(NULL)
  }
  
  encoded <- encode_train_test_numeric(df_train, df_test, target_train, target_test)
  df_train2 <- encoded$train
  df_test2 <- encoded$test
  
  Y_train <- df_train2$.target
  Y_test <- df_test2$.target
  
  X_train <- df_train2[, setdiff(names(df_train2), ".target"), drop = FALSE]
  X_test <- df_test2[, setdiff(names(df_test2), ".target"), drop = FALSE]
  
  if (ncol(X_train) == 0) {
    cat("Predictor bulunamadi.\n")
    return(NULL)
  }
  
  imp <- impute_by_train_median(X_train, X_test)
  X_train <- imp$X_train
  X_test <- imp$X_test
  
  if (any(is.na(Y_train)) || any(is.na(Y_test))) {
    cat("Target degiskeninde NA var. Dataset atlaniyor.\n")
    return(NULL)
  }
  
  scaled <- scale_train_test(X_train, X_test, Y_train, Y_test)
  
  list(
    X_train = scaled$X_train,
    X_test = scaled$X_test,
    Y_train = scaled$Y_train,
    Y_test = scaled$Y_test,
    target_name_train = target_train,
    target_name_test = target_test
  )
}

############################
# 19. MODEL RUNNER
############################
run_cart_frt <- function(X_train, X_test, Y_train, Y_test,
                         fuzzy.var = 0.05,
                         defuzz_k = 5,
                         defuzz_m = 1,
                         defuzz_threshold = 2,
                         frt_eps_f = 0.35,
                         frt_tau_supp = 0.03,
                         frt_tau_conf = 0.06,
                         frt_max_iter = 10,
                         frt_accept_metric = "MSE",
                         validation_fraction = 0.20,
                         validation_seed = 123,
                         min_rule_cases = 3,
                         max_rule_span = Inf,
                         flip_penalty = 0.001,
                         verbose = TRUE) {
  
  out <- list()
  
  df_train_model <- data.frame(X_train, Target = Y_train)
  tree0 <- build_crisp_tree(df_train_model, y_col = "Target")
  
  y_cart_train <- predict(tree0, X_train)
  y_cart_test <- predict(tree0, X_test)
  
  cart_train_err <- calc_crisp_errors(Y_train, y_cart_train)
  cart_test_err <- calc_crisp_errors(Y_test, y_cart_test)
  
  GFN_X_train <- lapply(X_train, fuzzify, fuzzy.var)
  GFN_Y_train <- fuzzify(Y_train, fuzzy.var)
  GFN_X_test <- lapply(X_test, fuzzify, fuzzy.var)
  GFN_Y_test <- fuzzify(Y_test, fuzzy.var)
  
  crisp_train_pred <- predict_tree_gfn(tree0, X_train, GFN_Y_train)
  crisp_test_routed <- route_all(tree0, X_test)
  
  Y_pred0_test <- predict_gfn_from_leaves(
    crisp_test_routed$leaf,
    crisp_train_pred$leaf_proto,
    crisp_train_pred$node_proto,
    crisp_train_pred$global_proto
  )
  
  initial_train_fuzzy <- calc_all_fuzzy_errors(
    GFN_Y_train, crisp_train_pred$pred,
    k = defuzz_k, m = defuzz_m, symmetry.threshold = defuzz_threshold
  )
  
  initial_test_fuzzy <- calc_all_fuzzy_errors(
    GFN_Y_test, Y_pred0_test,
    k = defuzz_k, m = defuzz_m, symmetry.threshold = defuzz_threshold
  )
  
  res_fuzzy <- update_tree_assignments_casewise(
    tree = tree0,
    X_df = X_train,
    GFN_X_list = GFN_X_train,
    GFN_Y = GFN_Y_train,
    y_true_crisp = Y_train,
    eps_f = frt_eps_f,
    tau_supp = frt_tau_supp,
    tau_conf = frt_tau_conf,
    max_iter = frt_max_iter,
    accept_metric = frt_accept_metric,
    defuzz_k = defuzz_k,
    defuzz_m = defuzz_m,
    defuzz_threshold = defuzz_threshold,
    validation_fraction = validation_fraction,
    validation_seed = validation_seed,
    min_rule_cases = min_rule_cases,
    max_rule_span = max_rule_span,
    flip_penalty = flip_penalty,
    verbose = verbose
  )
  
  routed_train_final <- route_all_with_rules(
    tree = tree0,
    X_df = X_train,
    GFN_X_list = GFN_X_train,
    flip_rules = res_fuzzy$flip_rules
  )
  
  routed_test_final <- route_all_with_rules(
    tree = tree0,
    X_df = X_test,
    GFN_X_list = GFN_X_test,
    flip_rules = res_fuzzy$flip_rules
  )
  
  leaf_proto_train_final <- compute_leaf_prototypes(routed_train_final$leaf, GFN_Y_train)
  node_proto_train_final <- compute_node_prototypes(routed_train_final$leaf, GFN_Y_train)
  global_proto_train <- GFN.mean_of_rows(GFN_Y_train)
  
  Y_pred_train_final <- predict_gfn_from_leaves(
    routed_train_final$leaf,
    leaf_proto_train_final,
    node_proto_train_final,
    global_proto_train
  )
  
  Y_pred_test_final <- predict_gfn_from_leaves(
    routed_test_final$leaf,
    leaf_proto_train_final,
    node_proto_train_final,
    global_proto_train
  )
  
  final_train_fuzzy <- calc_all_fuzzy_errors(
    GFN_Y_train, Y_pred_train_final,
    k = defuzz_k, m = defuzz_m, symmetry.threshold = defuzz_threshold
  )
  
  final_test_fuzzy <- calc_all_fuzzy_errors(
    GFN_Y_test, Y_pred_test_final,
    k = defuzz_k, m = defuzz_m, symmetry.threshold = defuzz_threshold
  )
  
  fuzzy_train_crisp <- apply(Y_pred_train_final, 1, function(gfn) {
    defuzzify(gfn, k = defuzz_k, m = defuzz_m, symmetry.threshold = defuzz_threshold)
  })
  
  fuzzy_test_crisp <- apply(Y_pred_test_final, 1, function(gfn) {
    defuzzify(gfn, k = defuzz_k, m = defuzz_m, symmetry.threshold = defuzz_threshold)
  })
  
  fuzzy_train_err <- calc_crisp_errors(Y_train, fuzzy_train_crisp)
  fuzzy_test_err <- calc_crisp_errors(Y_test, fuzzy_test_crisp)
  
  out$cart <- list(
    train_err = cart_train_err,
    test_err = cart_test_err,
    fuzzy_train_initial = initial_train_fuzzy,
    fuzzy_test_initial = initial_test_fuzzy
  )
  
  out$fuzzy <- list(
    train_final = final_train_fuzzy,
    test_final = final_test_fuzzy,
    flips = length(unlist(res_fuzzy$flips)),
    train_err = fuzzy_train_err,
    test_err = fuzzy_test_err,
    score_history = res_fuzzy$score_history,
    accept_metric = res_fuzzy$accept_metric,
    flip_rules = res_fuzzy$flip_rules,
    accepted_cases = res_fuzzy$accepted_cases
  )
  
  out
}
