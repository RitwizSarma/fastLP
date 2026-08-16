# Select the dataset here, then source or run this file from RStudio/R GUI.
dataset <- "n5000_t40"
repetitions <- 200L

if (!requireNamespace("fixest", quietly = TRUE)) {
  stop("Install fixest first: install.packages('fixest')", call. = FALSE)
}

config <- list(
  n5000_t40 = list(horizon = 12L),
  n10000_t40 = list(horizon = 12L)
)
if (!dataset %in% names(config)) {
  stop("Unknown dataset: ", dataset, call. = FALSE)
}
if (repetitions < 2L) {
  stop("repetitions must be at least 2", call. = FALSE)
}

# Resolve paths from the project root when run interactively, or from the file
# location when the script is sourced by another R file.
source_file <- tryCatch(sys.frames()[[1]]$ofile, error = function(...) NULL)
if (!is.null(source_file)) {
  benchmark_dir <- normalizePath(file.path(dirname(source_file), ".."))
} else {
  benchmark_dir <- normalizePath(file.path(getwd(), "benchmark"))
}
input_file <- file.path(benchmark_dir, "data", paste0(dataset, ".csv"))
output_file <- file.path(benchmark_dir, paste0("results_fixest_", dataset, ".csv"))
estimates_file <- file.path(benchmark_dir, paste0("estimates_fixest_", dataset, ".csv"))
if (!file.exists(input_file)) {
  stop("Dataset not found: ", input_file, ". Run benchmark/data/generate_data.py first.", call. = FALSE)
}

data <- read.csv(input_file)
data <- data[order(data$unit, data$time), ]
horizon_max <- config[[dataset]]$horizon

run_estimation <- function() {
  fits <- vector("list", horizon_max + 1L)
  for (h in 0:horizon_max) {
    # fixest constructs the estimation sample for each lead internally. No
    # manual sample-alignment code is needed for these complete panels.
    fits[[h + 1L]] <- fixest::feols(
      stats::as.formula(sprintf("f(outcome, %d) ~ shock + control | unit + time", h)),
      data = data,
      panel.id = ~unit + time,
      panel.time.step = "unitary",
      vcov = ~unit
    )
  }
  fits
}

timings <- vector("list", repetitions)
estimates <- NULL
for (rep in seq_len(repetitions)) {
  started <- proc.time()
  fits <- run_estimation()
  elapsed <- proc.time() - started
  if (rep == 1L) {
    estimates <- do.call(rbind, lapply(seq_along(fits), function(index) {
      coefficients <- stats::coef(fits[[index]])
      data.frame(
        dataset = dataset,
        horizon = index - 1L,
        estimate_shock = unname(coefficients[["shock"]]),
        estimate_control = unname(coefficients[["control"]])
      )
    }))
  }
  timings[[rep]] <- data.frame(
    dataset = dataset,
    repetition = rep,
    wall_seconds = unname(elapsed[["elapsed"]]),
    user_seconds = unname(elapsed[["user.self"]]),
    system_seconds = unname(elapsed[["sys.self"]])
  )
}
timings <- do.call(rbind, timings)
write.csv(timings, output_file, row.names = FALSE)
write.csv(estimates, estimates_file, row.names = FALSE)

summary_stats <- function(values) {
  c(
    mean = mean(values),
    sd = stats::sd(values),
    median = stats::median(values),
    q1 = unname(stats::quantile(values, 0.25)),
    q3 = unname(stats::quantile(values, 0.75)),
    min = min(values),
    max = max(values)
  )
}

cat(sprintf("Dataset: %s\n", dataset))
cat(sprintf("Rows: %s | N=%s | T=%s\n", format(nrow(data), big.mark = ","),
            length(unique(data$unit)), length(unique(data$time))))
cat(sprintf("Repetitions: %d\n", repetitions))
cat("Timing summary (seconds):\n")
summary_table <- rbind(
  wall = summary_stats(timings$wall_seconds),
  user = summary_stats(timings$user_seconds),
  system = summary_stats(timings$system_seconds)
)
print(summary_table)
cat(sprintf("Results written to: %s\n", output_file))
cat(sprintf("Estimates written to: %s\n", estimates_file))
