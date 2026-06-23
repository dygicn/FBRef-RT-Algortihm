source("bagging_main_frt_pipeline_functions.R")

build_crisp_tree_custom <- function(df, y_col, cp, minsplit, minbucket, maxdepth, xval = 0) {
  rpart::rpart(
    as.formula(paste(y_col, "~ .")),
    data = df,
    method = "anova",
    control = rpart::rpart.control(
      cp = cp,
      minsplit = minsplit,
      minbucket = minbucket,
      maxdepth = maxdepth,
      xval = xval
    )
  )
}

aggregate_numeric_predictions <- function(pred_list) {
  Reduce(`+`, pred_list) / length(pred_list)
}

aggregate_gfn_predictions <- function(pred_list) {
  n_obs <- nrow(pred_list[[1]])
  out <- matrix(NA_real_, nrow = n_obs, ncol = 2)
  colnames(out) <- c("Mean", "Variance")
  
  for (i in seq_len(n_obs)) {
    obs_preds <- do.call(rbind, lapply(pred_list, function(mat) mat[i, , drop = FALSE]))
    out[i, ] <- GFN.mean_of_rows(obs_preds)
  }
  
  out
}

fit_single_tree_frt_predict <- function(
    X_fit, Y_fit, X_eval_train, X_eval_test,
    cart_cp,
    cart_minsplit,
    cart_minbucket,
    cart_maxdepth = 10,
    frt_eps_f = 0.35,
    frt_tau_supp = 0.03,
    frt_tau_conf = 0.06,
    frt_max_iter = 10,
    frt_accept_metric = "MSE",
    fuzzy.var = 0.05,
    defuzz_k = 5,
    defuzz_m = 1,
    defuzz_threshold = 2,
    validation_fraction = 0.20,
    validation_seed = 123,
    min_rule_cases = 3,
    max_rule_span = Inf,
    flip_penalty = 0.001,
    verbose = FALSE) {
  
  df_fit_model <- data.frame(X_fit, Target = Y_fit)
  tree0 <- build_crisp_tree_custom(
    df = df_fit_model,
    y_col = "Target",
    cp = cart_cp,
    minsplit = cart_minsplit,
    minbucket = cart_minbucket,
    maxdepth = cart_maxdepth,
    xval = 0
  )
  
  cart_pred_train <- as.numeric(predict(tree0, X_eval_train))
  cart_pred_test <- as.numeric(predict(tree0, X_eval_test))
  
  GFN_X_fit <- lapply(X_fit, fuzzify, fuzzy.var)
  GFN_Y_fit <- fuzzify(Y_fit, fuzzy.var)
  GFN_X_eval_train <- lapply(X_eval_train, fuzzify, fuzzy.var)
  GFN_X_eval_test <- lapply(X_eval_test, fuzzify, fuzzy.var)
  
  routed_train_initial <- route_all(tree0, X_eval_train)
  routed_test_initial <- route_all(tree0, X_eval_test)
  
  leaf_proto_initial <- compute_leaf_prototypes(routed_train_initial$leaf, GFN_Y_fit)
  node_proto_initial <- compute_node_prototypes(routed_train_initial$leaf, GFN_Y_fit)
  global_proto_initial <- GFN.mean_of_rows(GFN_Y_fit)
  
  pred_train_initial <- predict_gfn_from_leaves(
    routed_train_initial$leaf,
    leaf_proto_initial,
    node_proto_initial,
    global_proto_initial
  )
  
  pred_test_initial <- predict_gfn_from_leaves(
    routed_test_initial$leaf,
    leaf_proto_initial,
    node_proto_initial,
    global_proto_initial
  )
  
  res_fuzzy <- update_tree_assignments_casewise(
    tree = tree0,
    X_df = X_fit,
    GFN_X_list = GFN_X_fit,
    GFN_Y = GFN_Y_fit,
    y_true_crisp = Y_fit,
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
  
  routed_fit_final <- route_all_with_rules(
    tree = tree0,
    X_df = X_fit,
    GFN_X_list = GFN_X_fit,
    flip_rules = res_fuzzy$flip_rules
  )
  
  routed_train_final <- route_all_with_rules(
    tree = tree0,
    X_df = X_eval_train,
    GFN_X_list = GFN_X_eval_train,
    flip_rules = res_fuzzy$flip_rules
  )
  
  routed_test_final <- route_all_with_rules(
    tree = tree0,
    X_df = X_eval_test,
    GFN_X_list = GFN_X_eval_test,
    flip_rules = res_fuzzy$flip_rules
  )
  
  leaf_proto_final <- compute_leaf_prototypes(routed_fit_final$leaf, GFN_Y_fit)
  node_proto_final <- compute_node_prototypes(routed_fit_final$leaf, GFN_Y_fit)
  global_proto_final <- GFN.mean_of_rows(GFN_Y_fit)
  
  pred_train_final <- predict_gfn_from_leaves(
    routed_train_final$leaf,
    leaf_proto_final,
    node_proto_final,
    global_proto_final
  )
  
  pred_test_final <- predict_gfn_from_leaves(
    routed_test_final$leaf,
    leaf_proto_final,
    node_proto_final,
    global_proto_final
  )
  
  list(
    cart_pred_train = cart_pred_train,
    cart_pred_test = cart_pred_test,
    gfn_pred_train_initial = pred_train_initial,
    gfn_pred_test_initial = pred_test_initial,
    gfn_pred_train_final = pred_train_final,
    gfn_pred_test_final = pred_test_final,
    flips = length(unlist(res_fuzzy$flips)),
    flip_rules = res_fuzzy$flip_rules
  )
}

run_bagging_frt_configurable <- function(
    X_train, X_test, Y_train, Y_test,
    bag_ntree = 25,
    bag_sample_frac = 1.0,
    cart_cp,
    cart_minsplit,
    cart_minbucket,
    cart_maxdepth = 10,
    frt_eps_f = 0.35,
    frt_tau_supp = 0.03,
    frt_tau_conf = 0.06,
    frt_max_iter = 10,
    frt_accept_metric = "MSE",
    fuzzy.var = 0.05,
    defuzz_k = 5,
    defuzz_m = 1,
    defuzz_threshold = 2,
    validation_fraction = 0.20,
    validation_seed = 123,
    min_rule_cases = 3,
    max_rule_span = Inf,
    flip_penalty = 0.001,
    verbose = FALSE) {
  
  n_train <- nrow(X_train)
  bag_size <- max(2, floor(n_train * bag_sample_frac))
  
  cart_train_preds <- list()
  cart_test_preds <- list()
  gfn_train_initial_list <- list()
  gfn_test_initial_list <- list()
  gfn_train_final_list <- list()
  gfn_test_final_list <- list()
  flips_per_tree <- numeric(bag_ntree)
  
  for (b in seq_len(bag_ntree)) {
    boot_idx <- sample(seq_len(n_train), size = bag_size, replace = TRUE)
    
    tree_res <- fit_single_tree_frt_predict(
      X_fit = X_train[boot_idx, , drop = FALSE],
      Y_fit = Y_train[boot_idx],
      X_eval_train = X_train,
      X_eval_test = X_test,
      cart_cp = cart_cp,
      cart_minsplit = cart_minsplit,
      cart_minbucket = cart_minbucket,
      cart_maxdepth = cart_maxdepth,
      frt_eps_f = frt_eps_f,
      frt_tau_supp = frt_tau_supp,
      frt_tau_conf = frt_tau_conf,
      frt_max_iter = frt_max_iter,
      frt_accept_metric = frt_accept_metric,
      fuzzy.var = fuzzy.var,
      defuzz_k = defuzz_k,
      defuzz_m = defuzz_m,
      defuzz_threshold = defuzz_threshold,
      validation_fraction = validation_fraction,
      validation_seed = validation_seed + b - 1,
      min_rule_cases = min_rule_cases,
      max_rule_span = max_rule_span,
      flip_penalty = flip_penalty,
      verbose = verbose
    )
    
    cart_train_preds[[b]] <- tree_res$cart_pred_train
    cart_test_preds[[b]] <- tree_res$cart_pred_test
    gfn_train_initial_list[[b]] <- tree_res$gfn_pred_train_initial
    gfn_test_initial_list[[b]] <- tree_res$gfn_pred_test_initial
    gfn_train_final_list[[b]] <- tree_res$gfn_pred_train_final
    gfn_test_final_list[[b]] <- tree_res$gfn_pred_test_final
    flips_per_tree[b] <- tree_res$flips
  }
  
  y_cart_train <- aggregate_numeric_predictions(cart_train_preds)
  y_cart_test <- aggregate_numeric_predictions(cart_test_preds)
  
  gfn_pred_train_initial <- aggregate_gfn_predictions(gfn_train_initial_list)
  gfn_pred_test_initial <- aggregate_gfn_predictions(gfn_test_initial_list)
  gfn_pred_train_final <- aggregate_gfn_predictions(gfn_train_final_list)
  gfn_pred_test_final <- aggregate_gfn_predictions(gfn_test_final_list)
  
  GFN_Y_train <- fuzzify(Y_train, fuzzy.var)
  GFN_Y_test <- fuzzify(Y_test, fuzzy.var)
  
  cart_train_err <- calc_crisp_errors(Y_train, y_cart_train)
  cart_test_err <- calc_crisp_errors(Y_test, y_cart_test)
  
  initial_fuzzy_train_scores <- calc_all_fuzzy_errors(
    GFN_Y_train, gfn_pred_train_initial,
    k = defuzz_k, m = defuzz_m, symmetry.threshold = defuzz_threshold
  )
  
  initial_fuzzy_test_scores <- calc_all_fuzzy_errors(
    GFN_Y_test, gfn_pred_test_initial,
    k = defuzz_k, m = defuzz_m, symmetry.threshold = defuzz_threshold
  )
  
  final_fuzzy_train_scores <- calc_all_fuzzy_errors(
    GFN_Y_train, gfn_pred_train_final,
    k = defuzz_k, m = defuzz_m, symmetry.threshold = defuzz_threshold
  )
  
  final_fuzzy_test_scores <- calc_all_fuzzy_errors(
    GFN_Y_test, gfn_pred_test_final,
    k = defuzz_k, m = defuzz_m, symmetry.threshold = defuzz_threshold
  )
  
  fuzzy_train_crisp <- apply(gfn_pred_train_final, 1, function(gfn) {
    defuzzify(gfn, k = defuzz_k, m = defuzz_m, symmetry.threshold = defuzz_threshold)
  })
  
  fuzzy_test_crisp <- apply(gfn_pred_test_final, 1, function(gfn) {
    defuzzify(gfn, k = defuzz_k, m = defuzz_m, symmetry.threshold = defuzz_threshold)
  })
  
  frt_crisp_train_err <- calc_crisp_errors(Y_train, fuzzy_train_crisp)
  frt_crisp_test_err <- calc_crisp_errors(Y_test, fuzzy_test_crisp)
  
  list(
    config = list(
      bag_ntree = bag_ntree,
      bag_sample_frac = bag_sample_frac,
      cart_cp = cart_cp,
      cart_minsplit = cart_minsplit,
      cart_minbucket = cart_minbucket,
      cart_maxdepth = cart_maxdepth,
      frt_eps_f = frt_eps_f,
      frt_tau_supp = frt_tau_supp,
      frt_tau_conf = frt_tau_conf,
      frt_max_iter = frt_max_iter,
      frt_accept_metric = frt_accept_metric,
      fuzzy.var = fuzzy.var,
      defuzz_k = defuzz_k,
      defuzz_m = defuzz_m,
      defuzz_threshold = defuzz_threshold,
      validation_fraction = validation_fraction,
      validation_seed = validation_seed,
      min_rule_cases = min_rule_cases,
      max_rule_span = max_rule_span,
      flip_penalty = flip_penalty
    ),
    cart = list(
      train_err = cart_train_err,
      test_err = cart_test_err,
      fuzzy_train_initial = initial_fuzzy_train_scores,
      fuzzy_test_initial = initial_fuzzy_test_scores
    ),
    fuzzy = list(
      flips = sum(flips_per_tree),
      avg_flips_per_tree = mean(flips_per_tree),
      train_final = final_fuzzy_train_scores,
      test_final = final_fuzzy_test_scores,
      crisp_train_err = frt_crisp_train_err,
      crisp_test_err = frt_crisp_test_err
    )
  )
}

sample_random_config <- function() {
  cart_minsplit <- sample(7:9, 1)
  
  list(
    bag_ntree = sample(c(15, 25, 35), 1),
    bag_sample_frac = sample(c(0.80, 1.00), 1),
    cart_cp = 10^runif(1, log10(0.0007), log10(0.0015)),
    cart_minsplit = cart_minsplit,
    cart_minbucket = 2,
    cart_maxdepth = sample(6:8, 1),
    frt_eps_f = runif(1, 0.09, 0.14),
    frt_tau_supp = runif(1, 0.05, 0.08),
    frt_tau_conf = runif(1, 0.18, 0.24),
    frt_max_iter = sample(4:6, 1),
    frt_accept_metric = "MSE",
    fuzzy.var = sample(c(0.03, 0.05, 0.07), 1),
    defuzz_k = sample(6:8, 1),
    defuzz_m = sample(c(1.0, 1.25, 1.5), 1),
    defuzz_threshold = 1,
    validation_fraction = 0.15,
    validation_seed = sample(100:999, 1),
    min_rule_cases = sample(1:2, 1),
    max_rule_span = sample(c(0.50, 0.75, Inf), 1),
    flip_penalty = sample(c(0, 0.0005), 1)
  )
}

evaluate_config_on_all_datasets <- function(
    cfg,
    train_dir = "train_data",
    test_dir = "test_data",
    verbose = FALSE) {
  
  train_files <- list.files(train_dir, pattern = "\\.csv$", full.names = TRUE)
  if (!length(train_files)) stop("No train csv files found in train_dir.")
  
  detail_rows <- list()
  
  for (train_file in train_files) {
    dataset_name <- basename(train_file)
    test_file <- file.path(test_dir, dataset_name)
    if (!file.exists(test_file)) next
    
    df_train <- read.csv(train_file)
    df_test <- read.csv(test_file)
    
    prep <- prepare_train_test_data(df_train, df_test)
    if (is.null(prep)) next
    
    res <- tryCatch(
      do.call(
        run_bagging_frt_configurable,
        c(
          list(
            X_train = prep$X_train,
            X_test = prep$X_test,
            Y_train = prep$Y_train,
            Y_test = prep$Y_test,
            verbose = verbose
          ),
          cfg
        )
      ),
      error = function(e) NULL
    )
    
    if (is.null(res)) next
    
    detail_rows[[length(detail_rows) + 1]] <- data.frame(
      dataset = dataset_name,
      n_train = nrow(prep$X_train),
      n_test = nrow(prep$X_test),
      bag_ntree = cfg$bag_ntree,
      bag_sample_frac = cfg$bag_sample_frac,
      cart_cp = cfg$cart_cp,
      cart_minsplit = cfg$cart_minsplit,
      cart_minbucket = cfg$cart_minbucket,
      cart_maxdepth = cfg$cart_maxdepth,
      frt_eps_f = cfg$frt_eps_f,
      frt_tau_supp = cfg$frt_tau_supp,
      frt_tau_conf = cfg$frt_tau_conf,
      frt_max_iter = cfg$frt_max_iter,
      frt_accept_metric = cfg$frt_accept_metric,
      fuzzy.var = cfg$fuzzy.var,
      defuzz_k = cfg$defuzz_k,
      defuzz_m = cfg$defuzz_m,
      defuzz_threshold = cfg$defuzz_threshold,
      validation_fraction = cfg$validation_fraction,
      validation_seed = cfg$validation_seed,
      min_rule_cases = cfg$min_rule_cases,
      max_rule_span = cfg$max_rule_span,
      flip_penalty = cfg$flip_penalty,
      flipnumber = res$fuzzy$flips,
      avg_flips_per_tree = res$fuzzy$avg_flips_per_tree,
      CART_MSE_train = res$cart$train_err$MSE,
      CART_RMSE_train = res$cart$train_err$RMSE,
      CART_MAE_train = res$cart$train_err$MAE,
      CART_MAPE_train = res$cart$train_err$MAPE,
      CART_MSE_test = res$cart$test_err$MSE,
      CART_RMSE_test = res$cart$test_err$RMSE,
      CART_MAE_test = res$cart$test_err$MAE,
      CART_MAPE_test = res$cart$test_err$MAPE,
      CART_init_FMSE_train = res$cart$fuzzy_train_initial$FMSE,
      CART_init_FMSE_test = res$cart$fuzzy_test_initial$FMSE,
      FRT_final_FMSE_train = res$fuzzy$train_final$FMSE,
      FRT_final_FMSE_test = res$fuzzy$test_final$FMSE,
      FRT_crisp_MSE_train = res$fuzzy$crisp_train_err$MSE,
      FRT_crisp_RMSE_train = res$fuzzy$crisp_train_err$RMSE,
      FRT_crisp_MAE_train = res$fuzzy$crisp_train_err$MAE,
      FRT_crisp_MAPE_train = res$fuzzy$crisp_train_err$MAPE,
      FRT_crisp_MSE_test = res$fuzzy$crisp_test_err$MSE,
      FRT_crisp_RMSE_test = res$fuzzy$crisp_test_err$RMSE,
      FRT_crisp_MAE_test = res$fuzzy$crisp_test_err$MAE,
      FRT_crisp_MAPE_test = res$fuzzy$crisp_test_err$MAPE,
      delta_FMSE_train = res$cart$fuzzy_train_initial$FMSE - res$fuzzy$train_final$FMSE,
      delta_FMSE_test = res$cart$fuzzy_test_initial$FMSE - res$fuzzy$test_final$FMSE,
      delta_MSE_train = res$cart$train_err$MSE - res$fuzzy$crisp_train_err$MSE,
      delta_MSE_test = res$cart$test_err$MSE - res$fuzzy$crisp_test_err$MSE,
      delta_RMSE_train = res$cart$train_err$RMSE - res$fuzzy$crisp_train_err$RMSE,
      delta_RMSE_test = res$cart$test_err$RMSE - res$fuzzy$crisp_test_err$RMSE,
      delta_MAE_train = res$cart$train_err$MAE - res$fuzzy$crisp_train_err$MAE,
      delta_MAE_test = res$cart$test_err$MAE - res$fuzzy$crisp_test_err$MAE,
      delta_MAPE_train = res$cart$train_err$MAPE - res$fuzzy$crisp_train_err$MAPE,
      delta_MAPE_test = res$cart$test_err$MAPE - res$fuzzy$crisp_test_err$MAPE,
      pct_change_FMSE_test = 100 * (res$cart$fuzzy_test_initial$FMSE - res$fuzzy$test_final$FMSE) /
        (res$cart$fuzzy_test_initial$FMSE + 1e-12),
      pct_change_MSE_test = 100 * (res$cart$test_err$MSE - res$fuzzy$crisp_test_err$MSE) /
        (res$cart$test_err$MSE + 1e-12),
      pct_change_RMSE_test = 100 * (res$cart$test_err$RMSE - res$fuzzy$crisp_test_err$RMSE) /
        (res$cart$test_err$RMSE + 1e-12),
      pct_change_MAE_test = 100 * (res$cart$test_err$MAE - res$fuzzy$crisp_test_err$MAE) /
        (res$cart$test_err$MAE + 1e-12),
      pct_change_MAPE_test = 100 * (res$cart$test_err$MAPE - res$fuzzy$crisp_test_err$MAPE) /
        (res$cart$test_err$MAPE + 1e-12)
    )
  }
  
  if (!length(detail_rows)) return(list(summary = NULL, details = NULL))
  
  details <- do.call(rbind, detail_rows)
  
  summary <- data.frame(
    bag_ntree = cfg$bag_ntree,
    bag_sample_frac = cfg$bag_sample_frac,
    cart_cp = cfg$cart_cp,
    cart_minsplit = cfg$cart_minsplit,
    cart_minbucket = cfg$cart_minbucket,
    cart_maxdepth = cfg$cart_maxdepth,
    frt_eps_f = cfg$frt_eps_f,
    frt_tau_supp = cfg$frt_tau_supp,
    frt_tau_conf = cfg$frt_tau_conf,
    frt_max_iter = cfg$frt_max_iter,
    frt_accept_metric = cfg$frt_accept_metric,
    fuzzy.var = cfg$fuzzy.var,
    defuzz_k = cfg$defuzz_k,
    defuzz_m = cfg$defuzz_m,
    defuzz_threshold = cfg$defuzz_threshold,
    validation_fraction = cfg$validation_fraction,
    validation_seed = cfg$validation_seed,
    min_rule_cases = cfg$min_rule_cases,
    max_rule_span = cfg$max_rule_span,
    flip_penalty = cfg$flip_penalty,
    n_datasets = nrow(details),
    improved_test_FMSE_count = sum(details$delta_FMSE_test > 0, na.rm = TRUE),
    improved_test_MSE_count = sum(details$delta_MSE_test > 0, na.rm = TRUE),
    avg_delta_FMSE_train = mean(details$delta_FMSE_train, na.rm = TRUE),
    avg_delta_FMSE_test = mean(details$delta_FMSE_test, na.rm = TRUE),
    avg_delta_MSE_test = mean(details$delta_MSE_test, na.rm = TRUE),
    avg_pct_change_FMSE_test = mean(details$pct_change_FMSE_test, na.rm = TRUE),
    avg_pct_change_MSE_test = mean(details$pct_change_MSE_test, na.rm = TRUE),
    avg_flipnumber = mean(details$flipnumber, na.rm = TRUE),
    avg_flips_per_tree = mean(details$avg_flips_per_tree, na.rm = TRUE)
  )
  
  summary$objective <- with(
    summary,
    0.55 * avg_delta_MSE_test +
      0.20 * avg_delta_FMSE_test +
      0.10 * avg_delta_FMSE_train +
      0.01 * improved_test_MSE_count +
      0.005 * improved_test_FMSE_count -
      0.001 * avg_flipnumber
  )
  
  list(summary = summary, details = details)
}

random_search_cart_frt <- function(
    n_iter = 50,
    train_dir = "train_data",
    test_dir = "test_data",
    seed = 123,
    verbose = FALSE,
    export_prefix = "bagging_random_search_cart_frt") {
  
  set.seed(seed)
  
  summary_rows <- list()
  detail_rows <- list()
  
  for (iter in seq_len(n_iter)) {
    cfg <- sample_random_config()
    
    if (verbose) {
      cat("\nBagging random search iteration:", iter, "of", n_iter, "\n")
    }
    
    eval_res <- evaluate_config_on_all_datasets(
      cfg = cfg,
      train_dir = train_dir,
      test_dir = test_dir,
      verbose = FALSE
    )
    
    if (is.null(eval_res$summary) || is.null(eval_res$details)) next
    
    eval_res$summary$iteration <- iter
    eval_res$details$iteration <- iter
    
    summary_rows[[length(summary_rows) + 1]] <- eval_res$summary
    detail_rows[[length(detail_rows) + 1]] <- eval_res$details
  }
  
  if (!length(summary_rows)) stop("Random search produced no valid results.")
  
  summary_df <- do.call(rbind, summary_rows)
  detail_df <- do.call(rbind, detail_rows)
  
  summary_df <- summary_df[
    order(
      -summary_df$objective,
      -summary_df$improved_test_MSE_count,
      -summary_df$improved_test_FMSE_count
    ),
  ]
  
  best_iteration <- summary_df$iteration[1]
  best_details <- detail_df[detail_df$iteration == best_iteration, ]
  
  openxlsx::write.xlsx(
    list(
      summary = summary_df,
      best_iteration_details = best_details,
      all_details = detail_df
    ),
    file = paste0(export_prefix, ".xlsx"),
    rowNames = FALSE
  )
  
  list(
    summary = summary_df,
    best_iteration_details = best_details,
    all_details = detail_df
  )
}

evaluate_config_on_single_dataset <- function(
    cfg,
    train_file,
    test_file,
    verbose = FALSE) {
  
  dataset_name <- basename(train_file)
  
  if (!file.exists(test_file)) {
    return(list(summary = NULL, details = NULL))
  }
  
  df_train <- read.csv(train_file)
  df_test <- read.csv(test_file)
  
  prep <- prepare_train_test_data(df_train, df_test)
  if (is.null(prep)) {
    return(list(summary = NULL, details = NULL))
  }
  
  res <- tryCatch(
    do.call(
      run_bagging_frt_configurable,
      c(
        list(
          X_train = prep$X_train,
          X_test = prep$X_test,
          Y_train = prep$Y_train,
          Y_test = prep$Y_test,
          verbose = verbose
        ),
        cfg
      )
    ),
    error = function(e) NULL
  )
  
  if (is.null(res)) {
    return(list(summary = NULL, details = NULL))
  }
  
  details <- data.frame(
    dataset = dataset_name,
    n_train = nrow(prep$X_train),
    n_test = nrow(prep$X_test),
    bag_ntree = cfg$bag_ntree,
    bag_sample_frac = cfg$bag_sample_frac,
    cart_cp = cfg$cart_cp,
    cart_minsplit = cfg$cart_minsplit,
    cart_minbucket = cfg$cart_minbucket,
    cart_maxdepth = cfg$cart_maxdepth,
    frt_eps_f = cfg$frt_eps_f,
    frt_tau_supp = cfg$frt_tau_supp,
    frt_tau_conf = cfg$frt_tau_conf,
    frt_max_iter = cfg$frt_max_iter,
    frt_accept_metric = cfg$frt_accept_metric,
    fuzzy.var = cfg$fuzzy.var,
    defuzz_k = cfg$defuzz_k,
    defuzz_m = cfg$defuzz_m,
    defuzz_threshold = cfg$defuzz_threshold,
    validation_fraction = cfg$validation_fraction,
    validation_seed = cfg$validation_seed,
    min_rule_cases = cfg$min_rule_cases,
    max_rule_span = cfg$max_rule_span,
    flip_penalty = cfg$flip_penalty,
    flipnumber = res$fuzzy$flips,
    avg_flips_per_tree = res$fuzzy$avg_flips_per_tree,
    CART_MSE_train = res$cart$train_err$MSE,
    CART_RMSE_train = res$cart$train_err$RMSE,
    CART_MAE_train = res$cart$train_err$MAE,
    CART_MAPE_train = res$cart$train_err$MAPE,
    CART_MSE_test = res$cart$test_err$MSE,
    CART_RMSE_test = res$cart$test_err$RMSE,
    CART_MAE_test = res$cart$test_err$MAE,
    CART_MAPE_test = res$cart$test_err$MAPE,
    CART_init_FMSE_train = res$cart$fuzzy_train_initial$FMSE,
    CART_init_FMSE_test = res$cart$fuzzy_test_initial$FMSE,
    FRT_final_FMSE_train = res$fuzzy$train_final$FMSE,
    FRT_final_FMSE_test = res$fuzzy$test_final$FMSE,
    FRT_crisp_MSE_train = res$fuzzy$crisp_train_err$MSE,
    FRT_crisp_RMSE_train = res$fuzzy$crisp_train_err$RMSE,
    FRT_crisp_MAE_train = res$fuzzy$crisp_train_err$MAE,
    FRT_crisp_MAPE_train = res$fuzzy$crisp_train_err$MAPE,
    FRT_crisp_MSE_test = res$fuzzy$crisp_test_err$MSE,
    FRT_crisp_RMSE_test = res$fuzzy$crisp_test_err$RMSE,
    FRT_crisp_MAE_test = res$fuzzy$crisp_test_err$MAE,
    FRT_crisp_MAPE_test = res$fuzzy$crisp_test_err$MAPE,
    delta_FMSE_train = res$cart$fuzzy_train_initial$FMSE - res$fuzzy$train_final$FMSE,
    delta_FMSE_test = res$cart$fuzzy_test_initial$FMSE - res$fuzzy$test_final$FMSE,
    delta_MSE_train = res$cart$train_err$MSE - res$fuzzy$crisp_train_err$MSE,
    delta_MSE_test = res$cart$test_err$MSE - res$fuzzy$crisp_test_err$MSE,
    delta_RMSE_train = res$cart$train_err$RMSE - res$fuzzy$crisp_train_err$RMSE,
    delta_RMSE_test = res$cart$test_err$RMSE - res$fuzzy$crisp_test_err$RMSE,
    delta_MAE_train = res$cart$train_err$MAE - res$fuzzy$crisp_train_err$MAE,
    delta_MAE_test = res$cart$test_err$MAE - res$fuzzy$crisp_test_err$MAE,
    delta_MAPE_train = res$cart$train_err$MAPE - res$fuzzy$crisp_train_err$MAPE,
    delta_MAPE_test = res$cart$test_err$MAPE - res$fuzzy$crisp_test_err$MAPE,
    pct_change_FMSE_test = 100 * (res$cart$fuzzy_test_initial$FMSE - res$fuzzy$test_final$FMSE) /
      (res$cart$fuzzy_test_initial$FMSE + 1e-12),
    pct_change_MSE_test = 100 * (res$cart$test_err$MSE - res$fuzzy$crisp_test_err$MSE) /
      (res$cart$test_err$MSE + 1e-12),
    pct_change_RMSE_test = 100 * (res$cart$test_err$RMSE - res$fuzzy$crisp_test_err$RMSE) /
      (res$cart$test_err$RMSE + 1e-12),
    pct_change_MAE_test = 100 * (res$cart$test_err$MAE - res$fuzzy$crisp_test_err$MAE) /
      (res$cart$test_err$MAE + 1e-12),
    pct_change_MAPE_test = 100 * (res$cart$test_err$MAPE - res$fuzzy$crisp_test_err$MAPE) /
      (res$cart$test_err$MAPE + 1e-12)
  )
  
  summary <- data.frame(
    bag_ntree = cfg$bag_ntree,
    bag_sample_frac = cfg$bag_sample_frac,
    cart_cp = cfg$cart_cp,
    cart_minsplit = cfg$cart_minsplit,
    cart_minbucket = cfg$cart_minbucket,
    cart_maxdepth = cfg$cart_maxdepth,
    frt_eps_f = cfg$frt_eps_f,
    frt_tau_supp = cfg$frt_tau_supp,
    frt_tau_conf = cfg$frt_tau_conf,
    frt_max_iter = cfg$frt_max_iter,
    frt_accept_metric = cfg$frt_accept_metric,
    fuzzy.var = cfg$fuzzy.var,
    defuzz_k = cfg$defuzz_k,
    defuzz_m = cfg$defuzz_m,
    defuzz_threshold = cfg$defuzz_threshold,
    validation_fraction = cfg$validation_fraction,
    validation_seed = cfg$validation_seed,
    min_rule_cases = cfg$min_rule_cases,
    max_rule_span = cfg$max_rule_span,
    flip_penalty = cfg$flip_penalty,
    n_datasets = 1,
    improved_test_FMSE_count = as.integer(details$delta_FMSE_test > 0),
    improved_test_MSE_count = as.integer(details$delta_MSE_test > 0),
    avg_delta_FMSE_train = details$delta_FMSE_train,
    avg_delta_FMSE_test = details$delta_FMSE_test,
    avg_delta_MSE_test = details$delta_MSE_test,
    avg_pct_change_FMSE_test = details$pct_change_FMSE_test,
    avg_pct_change_MSE_test = details$pct_change_MSE_test,
    avg_flipnumber = details$flipnumber,
    avg_flips_per_tree = details$avg_flips_per_tree
  )
  
  summary$objective <- with(
    summary,
    0.55 * avg_delta_MSE_test +
      0.20 * avg_delta_FMSE_test +
      0.10 * avg_delta_FMSE_train +
      0.01 * improved_test_MSE_count +
      0.005 * improved_test_FMSE_count -
      0.001 * avg_flipnumber
  )
  
  list(summary = summary, details = details)
}
