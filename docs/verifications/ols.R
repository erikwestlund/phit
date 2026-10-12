# Called by compare_ols.py; input CSV has y followed by x0, x1, ... .
args <- commandArgs(trailingOnly=TRUE)
d <- read.csv(args[1], na.strings='NA')
intercept <- any(vapply(d[-1], function(v) all(v[!is.na(v)] == 1), TRUE))
# Formula includes the explicit design columns and does not add another intercept.
fit <- lm(y ~ . - 1, data=d, na.action=na.omit)
s <- summary(fit)
n <- nobs(fit); p <- fit$rank; rss <- sum(resid(fit)^2)
x <- model.matrix(fit); y <- model.response(model.frame(fit))
# Explicitly match the project's intercept convention even though formula uses -1.
tss <- if(intercept) sum((y-mean(y))^2) else sum(y^2)
r2 <- 1-rss/tss
json <- function(x) {
  if(is.list(x)) {
    values <- vapply(x,json,'')
    if(!is.null(names(x))) return(paste0('{',paste(paste0('"',names(x),'":',values),collapse=','),'}'))
    return(paste0('[',paste(values,collapse=','),']'))
  }
  if(is.character(x)) return(paste0('"',x,'"'))
  if(length(x)!=1) return(json(unname(as.list(x))))
  if(!is.finite(x)) return('null')
  sprintf('%.17g',x)
}
rows <- function(m) unname(lapply(seq_len(nrow(m)),function(i) unname(as.list(m[i,]))))
result <- list(coefficients=unname(as.list(coef(fit))), covariance=rows(vcov(fit)),
  std_errors=unname(as.list(s$coefficients[,2])), t_statistics=unname(as.list(s$coefficients[,3])),
  p_values=unname(as.list(s$coefficients[,4])), conf_int=rows(confint(fit)),
  fitted=unname(as.list(fitted(fit))), residuals=unname(as.list(resid(fit))),
  rss=rss,residual_variance=rss/(n-p),sigma=sqrt(rss/(n-p)),r_squared=r2,
  adj_r_squared=1-(1-r2)*(n-as.integer(intercept))/(n-p),
  df_residual=n-p,log_likelihood=as.numeric(logLik(fit)))
cat(json(list(version=R.version.string,n_observations=n,result=result)))
