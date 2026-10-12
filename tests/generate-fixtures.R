# Run from repository root: Rscript tests/generate-fixtures.R
# Base R only; fixtures are runtime-independent.
json <- function(x) {
  if (is.list(x)) {
    values <- vapply(x, json, "")
    if (!is.null(names(x))) return(paste0('{', paste(paste0('"', names(x), '":', values), collapse=','), '}'))
    return(paste0('[', paste(values, collapse=','), ']'))
  }
  if (is.character(x)) return(paste0('"', x, '"'))
  if (length(x) != 1) return(paste0('[', paste(vapply(as.list(x), json, ""), collapse=','), ']'))
  if (!is.finite(x)) return('null')
  sprintf('%.17g', x)
}
rows <- function(m) unname(lapply(seq_len(nrow(m)), function(i) unname(as.list(m[i,]))))
z <- 1:10
housing <- read.csv("docs/data/housing.csv")
cases <- list(
  simulated_housing=list(x=cbind(1,housing$area_m2), y=housing$monthly_rent),
  simulated_housing_multiple=list(x=cbind(1,housing$area_m2,housing$bedrooms), y=housing$monthly_rent),
  simple=list(x=cbind(1,1:5), y=c(2.1,4.2,5.8,8.1,10.2)),
  multiple=list(x=cbind(1,z,c(3,1,4,2,5,0,2,4,1,3)), y=c(4,3,8,7,12,9,12,17,14,19)),
  no_intercept=list(x=cbind(z,z%%3), y=c(2,5,5,9,10,11,15,16,17,21)),
  correlated=list(x=cbind(1,z,z+c(0.01,-0.02,0.03,-0.01,0.02,-0.03,0.01,0.02,-0.01,0.03)), y=c(2,4,7,7,11,12,13,17,17,21)),
  perfect=list(x=cbind(1,1:5), y=2+3*(1:5)),
  zero=list(x=cbind(1,1:5), y=rep(0,5))
)
fixtures <- lapply(names(cases), function(name) {
  case <- cases[[name]]; x <- case$x; y <- case$y
  intercept <- any(apply(x,2,function(v) all(v==1)))
  # Formula explicitly includes/removes intercept; x excludes the supplied ones column.
  predictors <- if(intercept) x[,!apply(x,2,function(v) all(v==1)),drop=FALSE] else x
  fit <- if(intercept) lm(y ~ predictors) else lm(y ~ predictors - 1)
  s <- suppressWarnings(summary(fit)); ci <- suppressWarnings(confint(fit))
  expected <- list(covariance=rows(vcov(fit)), coefficients=as.list(coef(fit)), std_errors=as.list(s$coefficients[,2]),
    t_statistics=as.list(s$coefficients[,3]), p_values=as.list(s$coefficients[,4]),
    conf_int=rows(ci), fitted=as.list(fitted(fit)), residuals=as.list(resid(fit)),
    rss=sum(resid(fit)^2), residual_variance=sum(resid(fit)^2)/df.residual(fit),
    sigma=s$sigma, r_squared=s$r.squared, adj_r_squared=s$adj.r.squared,
    df_residual=df.residual(fit), log_likelihood=as.numeric(logLik(fit)))
  # Unname coefficient lists to serialize as arrays.
  expected <- lapply(expected, function(v) if(is.list(v)) unname(v) else v)
  list(name=name, x=rows(x), y=as.list(y), expected=expected)
})
writeLines(json(list(reference=R.version.string,cases=unname(fixtures))), 'tests/fixtures/ols-r.json')
