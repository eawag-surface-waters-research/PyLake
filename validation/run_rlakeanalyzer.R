args <- commandArgs(trailingOnly = TRUE)
input <- if (length(args) >= 1) args[[1]] else "validation/profiles.csv"
output <- if (length(args) >= 2) args[[2]] else "validation/rlakeanalyzer_results.csv"

if (!requireNamespace("rLakeAnalyzer", quietly = TRUE)) {
  stop("Install rLakeAnalyzer with install.packages('rLakeAnalyzer')")
}

profiles <- read.csv(input, stringsAsFactors = FALSE)
groups <- split(profiles, profiles$profile)

results <- lapply(names(groups), function(name) {
  current <- groups[[name]]
  value <- rLakeAnalyzer::thermo.depth(
    current$temperature,
    current$depth,
    seasonal = FALSE,
    mixed.cutoff = 1
  )
  data.frame(profile = name, r_thermocline = value)
})

write.csv(do.call(rbind, results), output, row.names = FALSE)
cat(output, "\n")
