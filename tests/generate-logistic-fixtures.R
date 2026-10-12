# Run from the repository root. Base R; runtime tests consume saved JSON only.
RNGkind("Mersenne-Twister", "Inversion", "Rejection")
set.seed(20261012)
n <- 80
hours <- round(runif(n, 1, 12), 2)
score <- round(pmin(95,pmax(20,45+3*hours+rnorm(n,0,8))),2)
probability <- plogis(-3.28+0.22*hours+0.018*score)
passed <- rbinom(n,1,probability)
observations <- data.frame(study_hours=hours, practice_score=score, passed=passed)
write.csv(observations,"docs/data/learning.csv",row.names=FALSE)
cases <- list(
  simulated_learning=list(x=cbind(1,hours,score),y=passed),
  simple=list(x=cbind(1,hours),y=passed),
  no_intercept=list(x=cbind(hours,score),y=passed),
  intercept_only=list(x=matrix(1,n,1),y=passed),
  correlated=list(x=cbind(1,hours,hours+0.2*sin(seq_len(n))),y=passed),
  imbalanced=list(x=cbind(1,hours,score),y=as.integer(seq_len(n) %in% c(5,23,57,74)))
)
json <- function(x) {
  if(is.list(x)) {
    values <- vapply(x,json,"")
    if(!is.null(names(x))) return(paste0('{',paste(paste0('"',names(x),'":',values),collapse=','),'}'))
    return(paste0('[',paste(values,collapse=','),']'))
  }
  if(is.character(x)) return(paste0('"',x,'"'))
  if(length(x)!=1) return(json(unname(as.list(x))))
  if(!is.finite(x)) stop("nonfinite reference value")
  sprintf('%.17g',x)
}
rows <- function(m) unname(lapply(seq_len(nrow(m)),function(i) unname(as.list(m[i,]))))
fixtures <- lapply(names(cases),function(name) {
  case <- cases[[name]]; x <- case$x; y <- case$y
  colnames(x) <- paste0('x',seq_len(ncol(x))-1)
  fit <- glm(y ~ x - 1,family=binomial(),control=glm.control(epsilon=1e-14,maxit=100))
  stopifnot(fit$converged)
  s <- summary(fit)
  intercept <- any(apply(x,2,function(v) all(v==1)))
  null_ll <- if(intercept) sum(dbinom(y,1,mean(y),log=TRUE)) else -length(y)*log(2)
  expected <- list(coefficients=unname(as.list(coef(fit))), covariance=rows(vcov(fit)),
    std_errors=unname(as.list(s$coefficients[,2])),z_statistics=unname(as.list(s$coefficients[,3])),
    p_values=unname(as.list(s$coefficients[,4])),conf_int=rows(confint.default(fit)),
    fitted=unname(as.list(fitted(fit))),residuals=unname(as.list(resid(fit,type='response'))),
    log_likelihood=as.numeric(logLik(fit)),deviance=deviance(fit),null_deviance=-2*null_ll,
    aic=AIC(fit),df_residual=df.residual(fit))
  list(name=name,x=rows(x),y=unname(as.list(y)),expected=expected)
})
writeLines(json(list(reference=R.version.string,cases=unname(fixtures))),"tests/fixtures/logistic-r.json")
