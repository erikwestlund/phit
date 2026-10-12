# Input CSV: y followed by explicit design columns. Reference uses Wald intervals.
args <- commandArgs(trailingOnly=TRUE)
d <- read.csv(args[1],na.strings='NA')
d <- d[complete.cases(d),,drop=FALSE]
fit <- glm(y ~ . - 1,data=d,family=binomial(),control=glm.control(epsilon=1e-14,maxit=100))
stopifnot(fit$converged)
s <- summary(fit)
intercept <- any(vapply(d[-1],function(v) all(v==1),TRUE))
null_ll <- if(intercept) sum(dbinom(d$y,1,mean(d$y),log=TRUE)) else -nrow(d)*log(2)
json <- function(x) {
  if(is.list(x)) {
    values <- vapply(x,json,'')
    if(!is.null(names(x))) return(paste0('{',paste(paste0('"',names(x),'":',values),collapse=','),'}'))
    return(paste0('[',paste(values,collapse=','),']'))
  }
  if(is.character(x)) return(paste0('"',x,'"'))
  if(length(x)!=1) return(json(unname(as.list(x))))
  if(!is.finite(x)) stop('nonfinite reference result')
  sprintf('%.17g',x)
}
rows <- function(m) unname(lapply(seq_len(nrow(m)),function(i) unname(as.list(m[i,]))))
result <- list(coefficients=unname(as.list(coef(fit))),covariance=rows(vcov(fit)),
  std_errors=unname(as.list(s$coefficients[,2])),z_statistics=unname(as.list(s$coefficients[,3])),
  p_values=unname(as.list(s$coefficients[,4])),conf_int=rows(confint.default(fit)),
  fitted=unname(as.list(fitted(fit))),residuals=unname(as.list(resid(fit,type='response'))),
  log_likelihood=as.numeric(logLik(fit)),deviance=deviance(fit),null_deviance=-2*null_ll,
  aic=AIC(fit),df_residual=df.residual(fit))
cat(json(list(version=R.version.string,n_observations=nobs(fit),result=result)))
