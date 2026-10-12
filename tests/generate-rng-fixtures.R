# R quantile references on independent Python-generated ChaCha20 uniforms.
args <- commandArgs(trailingOnly=TRUE)
data <- jsonlite::fromJSON(args[1], simplifyVector=FALSE)
expected <- list()
for (i in seq_along(data$cases)) {
  c <- data$cases[[i]]; u <- unlist(c$input); p <- c$params
  result <- switch(c$distribution, uniform=u,
    normal=qnorm(u,p$mean,p$sd), t=qt(u,p$df), chisquare=qchisq(u,p$df), f=qf(u,p$df1,p$df2),
    beta=qbeta(u,p$alpha,p$beta), gamma=qgamma(u,p$shape,scale=p$scale), exponential=qexp(u,p$rate),
    lognormal=qlnorm(u,p$meanlog,p$sdlog), logistic=qlogis(u,p$location,p$scale),
    binomial=qbinom(u,p$size,p$prob), poisson=qpois(u,p$lambda),
    nbinom=qnbinom(u,p$size,p$prob), nbinom_mu=qnbinom(u,p$size,mu=p$mu))
  stopifnot(all(is.finite(result)))
  expected[[i]] <- unname(result)
}
writeLines(jsonlite::toJSON(list(expected=expected,reference=R.version.string,
  jsonlite=as.character(packageVersion('jsonlite'))),auto_unbox=TRUE,digits=NA), args[2])
