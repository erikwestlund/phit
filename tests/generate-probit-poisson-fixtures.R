# Base R reference generation; saved observations/fixtures are runtime-independent.
RNGkind('Mersenne-Twister','Inversion','Rejection')
set.seed(20261013)
n <- 120
hours <- round(runif(n,1,12),2)
score <- round(pmin(95,pmax(20,45+3*hours+rnorm(n,0,8))),2)
passed <- as.integer(-1.9+0.13*hours+0.011*score+rnorm(n)>0)
write.csv(data.frame(study_hours=hours,practice_score=score,passed=passed),'docs/data/probit-learning.csv',row.names=FALSE)
probit <- list(simulated_learning=list(x=cbind(1,hours,score),y=passed),
 simple=list(x=cbind(1,hours),y=passed), no_intercept=list(x=cbind(hours,score),y=passed),
 intercept_only=list(x=matrix(1,n,1),y=passed),
 correlated=list(x=cbind(1,hours,hours+0.2*sin(seq_len(n))),y=passed),
 imbalanced=list(x=cbind(1,hours,score),y=as.integer(seq_len(n) %in% c(5,23,57,74))))
set.seed(20261014)
visitors <- round(runif(n,80,300))
staff <- round(pmin(10,pmax(2,visitors/50+rnorm(n,0,0.8))),2)
calls <- rpois(n,exp(0.1+0.005*visitors-0.12*staff))
write.csv(data.frame(visitors=visitors,staff=staff,calls=calls),'docs/data/clinic.csv',row.names=FALSE)
poisson <- list(simulated_clinic=list(x=cbind(1,visitors,staff),y=calls),
 simple=list(x=cbind(1,visitors),y=calls), no_intercept=list(x=cbind(visitors,staff),y=calls),
 intercept_only=list(x=matrix(1,n,1),y=calls),
 correlated=list(x=cbind(1,visitors,visitors+sin(seq_len(n))),y=calls),
 sparse=list(x=cbind(1,visitors,staff),y=rpois(n,0.2)))
json <- function(x) {
 if(is.list(x)) { values <- vapply(x,json,'');
  if(!is.null(names(x))) return(paste0('{',paste(paste0('"',names(x),'":',values),collapse=','),'}'))
  return(paste0('[',paste(values,collapse=','),']')) }
 if(is.character(x)) return(paste0('"',x,'"'))
 if(length(x)!=1) return(json(unname(as.list(x))))
 if(!is.finite(x)) stop('Nonfinite fixture'); sprintf('%.17g',x)
}
rows <- function(m) unname(lapply(seq_len(nrow(m)),function(i) unname(as.list(m[i,]))))
for(model in c('probit','poisson')) {
 cases <- get(model)
 fixtures <- lapply(names(cases),function(name) {
  case <- cases[[name]]; x <- case$x; y <- case$y
  colnames(x) <- paste0('x',seq_len(ncol(x))-1)
  family <- if(model=='probit') binomial(link='probit') else poisson()
  fit <- glm(y~x-1,family=family,control=glm.control(epsilon=1e-16,maxit=100))
  stopifnot(fit$converged)
  s <- summary(fit)
  intercept <- any(apply(x,2,function(v) all(v==1)))
  baseline <- if(intercept) mean(y) else if(model=='probit') 0.5 else 1
  nullfit <- if(model=='probit') sum(dbinom(y,1,baseline,log=TRUE)) else sum(dpois(y,baseline,log=TRUE))
  null_deviance <- if(model=='probit') -2*nullfit else sum(family$dev.resids(y,rep(baseline,length(y)),rep(1,length(y))))
  expected <- list(coefficients=unname(as.list(coef(fit))),covariance=rows(vcov(fit)),
   std_errors=unname(as.list(s$coefficients[,2])),z_statistics=unname(as.list(s$coefficients[,3])),
   p_values=unname(as.list(s$coefficients[,4])),conf_int=rows(confint.default(fit)),
   fitted=unname(as.list(fitted(fit))),residuals=unname(as.list(resid(fit,type='response'))),
   log_likelihood=as.numeric(logLik(fit)),deviance=deviance(fit),null_deviance=null_deviance,
   aic=AIC(fit),df_residual=df.residual(fit))
  list(name=name,x=rows(x),y=unname(as.list(y)),expected=expected)
 })
 writeLines(json(list(reference=R.version.string,cases=unname(fixtures))),paste0('tests/fixtures/',model,'-r.json'))
}
