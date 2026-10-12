args <- commandArgs(trailingOnly=TRUE)
start <- proc.time()[['elapsed']]
d <- read.csv(args[1])
load_seconds <- proc.time()[['elapsed']]-start
n <- nrow(d); p <- ncol(d)-1
indices <- unique(floor(seq(0,n-1,length.out=10)))+1
rows <- function(m) unname(lapply(seq_len(nrow(m)),function(i) unname(as.list(m[i,]))))
estimate <- function() {
  fit <- glm(y ~ . - 1,data=d,family=binomial(link='probit'),control=glm.control(epsilon=1e-16,maxit=100))
  stopifnot(fit$converged)
  s <- summary(fit)
  fitted <- fitted(fit); residuals <- resid(fit,type='response')
  null_ll <- sum(dbinom(d$y,1,mean(d$y),log=TRUE))
  list(coefficients=unname(as.list(coef(fit))),covariance=rows(vcov(fit)),
    std_errors=unname(as.list(s$coefficients[,2])),z_statistics=unname(as.list(s$coefficients[,3])),
    p_values=unname(as.list(s$coefficients[,4])),conf_int=rows(confint.default(fit)),
    log_likelihood=as.numeric(logLik(fit)),deviance=deviance(fit),null_deviance=-2*null_ll,
    aic=AIC(fit),df_residual=df.residual(fit),fitted_sample=unname(as.list(fitted[indices])),
    residuals_sample=unname(as.list(residuals[indices])))
}
invisible(estimate())
times <- numeric(as.integer(args[2]))
for(i in seq_along(times)) {
  gc()
  start <- proc.time()[['elapsed']]
  result <- estimate()
  times[i] <- proc.time()[['elapsed']]-start
}
json <- function(x) {
  if(is.list(x)) {
    values <- vapply(x,json,'')
    if(!is.null(names(x))) return(paste0('{',paste(paste0('"',names(x),'":',values),collapse=','),'}'))
    return(paste0('[',paste(values,collapse=','),']'))
  }
  if(is.character(x)) return(paste0('"',x,'"'))
  if(length(x)!=1) return(json(unname(as.list(x))))
  if(!is.finite(x)) stop('Nonfinite benchmark output')
  sprintf('%.17g',x)
}
cat(json(list(version=R.version.string,BLAS=unname(extSoftVersion()['BLAS']),load_seconds=load_seconds,
    fit_seconds=as.list(times),sample_indices=as.list(indices-1),result=result)))
