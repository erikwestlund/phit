args <- commandArgs(trailingOnly=TRUE)
start <- proc.time()[['elapsed']]
input <- jsonlite::fromJSON(args[1]); x <- input$values
calculate <- switch(input$operation,
  normal_cdf=function() pnorm(x), normal_sf=function() pnorm(x, lower.tail=FALSE),
  t_quantile=function() qt(x, 12), binomial_logpmf=function() dbinom(x, 100, .3, log=TRUE),
  poisson_logpmf=function() dpois(x, 12, log=TRUE), beta_quantile=function() qbeta(x, 2, 5),
  chisquare_cdf=function() pchisq(x, 1e6), mean=function() mean(x), variance=function() var(x))
load <- proc.time()[['elapsed']] - start
result <- calculate(); times <- list()
if (args[2] == 'time') {
  times <- replicate(as.integer(args[3]), {
    start <- proc.time()[['elapsed']]
    for (j in seq_len(input$batch)) result <- calculate()
    (proc.time()[['elapsed']] - start)/input$batch
  }, simplify=FALSE)
}
n <- length(result); indices <- floor((0:16)*(n-1)/16)+1
cat(jsonlite::toJSON(list(version=R.version.string, jsonlite=as.character(packageVersion('jsonlite')),
    load_seconds=load, seconds=times, result=list(length=n, sum=sum(result), sum_squares=sum(result*result),
    min=min(result), max=max(result), sample=unname(result[indices]))), auto_unbox=TRUE, digits=NA))
