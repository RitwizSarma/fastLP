#!/usr/bin/env Rscript

# Choose one: small_balanced, large_balanced, small_unbalanced, large_unbalanced.
dataset <- "small_balanced"

if (!requireNamespace("fixest", quietly = TRUE)) {
  stop("Install fixest first: install.packages('fixest')", call. = FALSE)
}

config <- list(
  small_balanced = list(horizon = 12L),
  small_unbalanced = list(horizon = 12L),
  large_balanced = list(horizon = 2L),
  large_unbalanced = list(horizon = 2L)
)
if (!dataset %in% names(config)) {
  stop("Unknown dataset: ", dataset, call. = FALSE)
}

# Resolve paths relative to this script, so it can be launched from any directory.
script_path <- normalizePath(sub("^--file=", "", commandArgs(trailingOnly = FALSE)[grep("^--file=", commandArgs(trailingOnly = FALSE))][1]))
benchmark_dir <- normalizePath(file.path(dirname(script_path), ".."))
input_file <- file.path(benchmark_dir, "data", paste0(dataset, ".csv"))
output_file <- file.path(benchmark_dir, "results_fixest_", dataset, ".csv")
if (!file.exists(input_file)) {
  stop("Dataset not found: ", input_file, ". Run benchmark/data/generate_data.py first.", call. = FALSE)
}

data <- read.csv(input_file)
data <- data[order(data$unit, data$time), ]
horizon_max <- config[[dataset]]$horizon

results <- vector("list", horizon_max + 1L)
calculation_start <- proc.time()[["elapsed"]]
# Construct the same exact-time common anchor sample used by fastLP. This is
# deliberately timed because fastLP's reported fit includes sample alignment.
observed_keys <- paste(data$unit, data$time, sep = "\r")
common_sample <- rep(TRUE, nrow(data))
for (h in 0:horizon_max) {
  future_keys <- paste(data$unit, data$time + h, sep = "\r")
  common_sample <- common_sample & future_keys %in% observed_keys
}
for (h in 0:horizon_max) {
  # f(outcome, h) is outcome at t+h. panel.id makes leads respect missing
  # periods, which is necessary for the unbalanced datasets.
  fit <- fixest::feols(
    stats::as.formula(sprintf("f(outcome, %d) ~ shock + control | unit + time", h)),
    data = data,
    subset = common_sample,
    panel.id = ~unit + time,
    panel.time.step = "unitary",
    vcov = ~unit
  )
  coefs <- fixest::coeftable(fit)
  results[[h + 1L]] <- data.frame(
    dataset = dataset,
    horizon = h,
    estimate = coefs["shock", "Estimate"],
    std_error = coefs["shock", "Std. Error"],
    n_obs = stats::nobs(fit)
  )
}
calculation_seconds <- proc.time()[["elapsed"]] - calculation_start

results <- do.call(rbind, results)
results$calculation_seconds <- calculation_seconds
write.csv(results, output_file, row.names = FALSE)
print(results)
cat(sprintf("\nfixest LP calculation time for %s: %.3f seconds\n", dataset, calculation_seconds))
cat(sprintf("Results written to: %s\n", output_file))
