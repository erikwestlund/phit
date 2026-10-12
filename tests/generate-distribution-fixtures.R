# Run from repository root: Rscript tests/generate-distribution-fixtures.R
# Base R only. Reference values for the v0.2 distribution and descriptive APIs.
# Parameter names match the PHP arguments; nonfinite numbers are written as strings.
number <- function(x) {
  if (is.nan(x)) return('"NaN"')
  if (is.infinite(x)) return(if (x > 0) '"Inf"' else '"-Inf"')
  sprintf('%.17g', x)
}
json <- function(x) {
  if (is.list(x)) {
    values <- vapply(x, json, "")
    if (!is.null(names(x))) return(paste0('{', paste(paste0('"', names(x), '":', values), collapse=','), '}'))
    return(paste0('[', paste(values, collapse=','), ']'))
  }
  if (is.character(x)) return(paste0('"', x, '"'))
  if (is.logical(x)) return(if (x) 'true' else 'false')
  if (length(x) == 1) return(number(x)) # Every array in these fixtures has at least two values.
  paste0('[', paste(vapply(x, number, ""), collapse=','), ']')
}

probabilities <- c(0, 1e-300, 1e-100, 1e-20, 1e-10, 0.001, 0.025, 0.1, 0.5, 0.9, 0.975, 0.999,
  1 - 1e-10, 1 - 2^-52, 1)

# Each distribution maps the PHP parameter list onto R's d/p/q functions.
continuous <- list(
  normal = list(d = dnorm, p = pnorm, q = qnorm, args = function(a) list(mean = a$mean, sd = a$sd),
    params = list(list(mean = 0, sd = 1), list(mean = 3, sd = 2), list(mean = 1e6, sd = 1e3), list(mean = -2, sd = 1e-8)),
    x = function(a) c(-Inf, a$mean + a$sd * c(-1e10, -40, -38.5, -37, -20, -10, -8.5, -8, -5, -1.96, -1, -1e-10, 0,
      1e-10, 0.5, 1, 1.96, 3, 8, 8.5, 10, 20, 37, 38.5, 40, 1e10), Inf)),
  t = list(d = dt, p = pt, q = qt, args = function(a) list(df = a$df),
    params = lapply(c(0.5, 1, 2, 3.5, 12, 30, 100, 1e4, 1e6), function(df) list(df = df)),
    x = function(a) c(-Inf, -1e10, -1e5, -100, -10, -2.5, -1, -0.1, 0, 1e-8, 0.1, 1, 2.1, 5, 30, 100, 1e5, 1e10, Inf)),
  chisquare = list(d = dchisq, p = pchisq, q = qchisq, args = function(a) list(df = a$df),
    params = lapply(c(0.5, 1, 2, 4, 5, 30, 1000, 1e6), function(df) list(df = df)),
    x = function(a) sort(unique(c(-1, 0, 1e-300, 1e-100, 1e-10, 1e-3, 0.5, 1, 12.4, 12.5, 100, 1000, 1e4,
      a$df * c(0.1, 0.5, 1, 2, 5), a$df + c(-5, 5, 10, 40) * sqrt(2 * a$df), 1e7, Inf)))),
  f = list(d = df, p = pf, q = qf, args = function(a) list(df1 = a$df1, df2 = a$df2),
    params = list(list(df1 = 4, df2 = 40), list(df1 = 1, df2 = 1), list(df1 = 0.5, df2 = 3), list(df1 = 10, df2 = 1e6),
      list(df1 = 100, df2 = 100), list(df1 = 5, df2 = 0.5), list(df1 = 2, df2 = 2)),
    x = function(a) c(-1, 0, 1e-200, 1e-10, 0.1, 0.5, 1, 2, 3.2, 10, 100, 1e5, 1e20, Inf)),
  beta = list(d = dbeta, p = pbeta, q = qbeta, args = function(a) list(shape1 = a$alpha, shape2 = a$beta),
    params = list(list(alpha = 2, beta = 5), list(alpha = 0.5, beta = 0.5), list(alpha = 1, beta = 1), list(alpha = 0.1, beta = 10),
      list(alpha = 50, beta = 200), list(alpha = 1000, beta = 1000), list(alpha = 3, beta = 1)),
    x = function(a) c(-0.5, 0, 1e-300, 1e-100, 1e-10, 0.01, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99, 1 - 1e-10, 1, 1.5)),
  gamma = list(d = dgamma, p = pgamma, q = qgamma, args = function(a) list(shape = a$shape, scale = a$scale),
    params = list(list(shape = 2, scale = 3), list(shape = 0.5, scale = 1), list(shape = 1, scale = 1), list(shape = 0.01, scale = 1),
      list(shape = 100, scale = 0.01), list(shape = 1e4, scale = 2), list(shape = 3, scale = 1e5)),
    x = function(a) c(-1, 0, a$scale * c(1e-300, 1e-50, 1e-8, 0.1, 1), a$scale * a$shape * c(0.5, 1, 2, 10),
      a$scale * (a$shape + c(-5, 5, 40) * sqrt(a$shape)), a$scale * c(1e3, 1e5) * max(1, a$shape), Inf)),
  exponential = list(d = dexp, p = pexp, q = qexp, args = function(a) list(rate = a$rate),
    params = lapply(c(1, 0.5, 1e-5, 1e5), function(rate) list(rate = rate)),
    x = function(a) c(-1, 0, c(1e-300, 1e-20, 1e-5, 0.5, 1, 10, 100, 700, 800, 1e5) / a$rate, Inf)),
  lognormal = list(d = dlnorm, p = plnorm, q = qlnorm, args = function(a) list(meanlog = a$meanlog, sdlog = a$sdlog),
    params = list(list(meanlog = 0, sdlog = 1), list(meanlog = 2, sdlog = 0.5), list(meanlog = -5, sdlog = 3)),
    x = function(a) c(-1, 0, 1e-300, exp(a$meanlog + a$sdlog * c(-40, -10, -2, -0.5, 0, 0.5, 2, 10, 40)), Inf)),
  logistic = list(d = dlogis, p = plogis, q = qlogis, args = function(a) list(location = a$location, scale = a$scale),
    params = list(list(location = 0, scale = 1), list(location = 5, scale = 2), list(location = 0, scale = 1e-5)),
    x = function(a) c(-Inf, a$location + a$scale * c(-800, -40, -10, -1, -1e-10, 0, 1e-10, 1, 10, 40, 800), Inf))
)

discrete <- list(
  binomial = list(d = dbinom, p = pbinom, q = qbinom, args = function(a) list(size = a$size, prob = a$prob),
    params = list(list(size = 10, prob = 0.5), list(size = 1, prob = 0.3), list(size = 100, prob = 0.01),
      list(size = 1000, prob = 0.999), list(size = 1e6, prob = 0.3), list(size = 50, prob = 0), list(size = 50, prob = 1),
      list(size = 0, prob = 0.4)),
    x = function(a) sort(unique(c(-1, 0, 1, 2.5, 3, 5, 7, floor(a$size / 2), floor(a$size * a$prob),
      a$size - 1, a$size, a$size + 1, 1e9)))),
  poisson = list(d = dpois, p = ppois, q = qpois, args = function(a) list(lambda = a$lambda),
    params = lapply(c(3.2, 1e-10, 0.5, 12, 85, 1e4, 1e6, 0), function(lambda) list(lambda = lambda)),
    x = function(a) sort(unique(c(-1, 0, 1, 2.5, 5, 15, 100, floor(a$lambda),
      pmax(0, floor(a$lambda + c(-10, 10, 40) * sqrt(a$lambda))), 1e5, 1e7, 1e15)))),
  nbinom = list(d = dnbinom, p = pnbinom, q = qnbinom, args = function(a) list(size = a$size, prob = a$prob),
    params = list(list(size = 3, prob = 0.4), list(size = 0.5, prob = 0.1), list(size = 100, prob = 0.99),
      list(size = 1e5, prob = 0.5), list(size = 2, prob = 1)),
    x = function(a) sort(unique(c(-1, 0, 1, 2.5, 4, 10, 50, 500, 1e4, 1e5, 1e6, 1e9)))),
  nbinom_mu = list(d = function(x, ...) dnbinom(x, ...), p = pnbinom, q = qnbinom,
    args = function(a) list(size = a$size, mu = a$mu),
    params = list(list(mu = 4.5, size = 2), list(mu = 10, size = 0.5), list(mu = 3, size = 1e6),
      list(mu = 1e-3, size = 10), list(mu = 1e5, size = 1), list(mu = 0, size = 3)),
    x = function(a) sort(unique(c(-1, 0, 1, 2.5, 4, 10, 50, 500, 1e4, 1e5, 1e6, 1e9))))
)

cases <- list()
add <- function(distribution, params, operation, input, expected) {
  cases[[length(cases) + 1]] <<- list(distribution = distribution, params = params, operation = operation,
    input = input, expected = expected)
}
for (family in list(list(set = continuous, density = 'pdf'), list(set = discrete, density = 'pmf'))) {
  for (name in names(family$set)) {
    d <- family$set[[name]]
    for (params in d$params) {
      a <- d$args(params); x <- d$x(params)
      call <- function(f, first, ...) suppressWarnings(do.call(f, c(list(first), a, list(...))))
      add(name, params, family$density, x, call(d$d, x))
      add(name, params, paste0('log', family$density), x, call(d$d, x, log = TRUE))
      add(name, params, 'cdf', x, call(d$p, x))
      add(name, params, 'logcdf', x, call(d$p, x, log.p = TRUE))
      add(name, params, 'sf', x, call(d$p, x, lower.tail = FALSE))
      add(name, params, 'logsf', x, call(d$p, x, lower.tail = FALSE, log.p = TRUE))
      add(name, params, 'quantile', probabilities, call(d$q, probabilities))
    }
  }
}

# Descriptive statistics: sample (n - 1) variance, type 7 quantiles, frequency weights.
set.seed(20261012)
descriptive_inputs <- list(
  small = c(2.5, 3.1, 4.7, 1.2, 9.8, 5.5, 3.3),
  integers = c(4, 8, 15, 16, 23, 42),
  offset = 1e9 + c(4, 7, 13, 16),
  tiny = c(1e-12, 3e-12, 2e-12, 5e-12),
  constant = rep(7.25, 5),
  normal = round(rnorm(200, mean = 50, sd = 12), 6),
  skewed = round(rexp(150, rate = 0.2), 6),
  pair = c(1, 2)
)
descriptive <- lapply(names(descriptive_inputs), function(name) {
  v <- descriptive_inputs[[name]]
  list(name = name, values = v, mean = mean(v), variance = var(v), sd = sd(v),
    population_variance = mean((v - mean(v))^2), median = median(v),
    quantile_probabilities = c(0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1),
    quantiles = unname(quantile(v, c(0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1), type = 7)))
})
x <- round(rnorm(80, 10, 3), 6); y <- round(2 * x + rnorm(80, 0, 4), 6)
weights <- round(runif(7, 0.2, 3), 6); values <- descriptive_inputs$small
frequency <- c(3, 1, 2, 5, 1, 2, 4)
weighted_variance <- function(v, w, type) {
  m <- sum(w * v) / sum(w); ss <- sum(w * (v - m)^2)
  switch(type, frequency = ss / (sum(w) - 1), reliability = ss / (sum(w) - sum(w^2) / sum(w)), population = ss / sum(w))
}
pairs <- list(
  list(name = 'correlated', x = x, y = y, covariance = cov(x, y), correlation = cor(x, y),
    population_covariance = cov(x, y) * (length(x) - 1) / length(x)),
  list(name = 'negative', x = c(1, 2, 3, 4, 5), y = c(10, 8, 6.5, 3, 1), covariance = cov(c(1, 2, 3, 4, 5), c(10, 8, 6.5, 3, 1)),
    correlation = cor(c(1, 2, 3, 4, 5), c(10, 8, 6.5, 3, 1)), population_covariance = cov(c(1, 2, 3, 4, 5), c(10, 8, 6.5, 3, 1)) * 4 / 5),
  list(name = 'offset', x = 1e9 + c(1, 2, 3, 4), y = 1e9 + c(2, 4, 5, 9), covariance = cov(1e9 + c(1, 2, 3, 4), 1e9 + c(2, 4, 5, 9)),
    correlation = cor(1e9 + c(1, 2, 3, 4), 1e9 + c(2, 4, 5, 9)), population_covariance = cov(1e9 + c(1, 2, 3, 4), 1e9 + c(2, 4, 5, 9)) * 3 / 4)
)
weighted <- list(
  list(name = 'analytic', values = values, weights = weights, mean = weighted.mean(values, weights),
    frequency = weighted_variance(values, weights, 'frequency'), reliability = weighted_variance(values, weights, 'reliability'),
    population = weighted_variance(values, weights, 'population'),
    cov_wt_unbiased = cov.wt(cbind(values), weights, method = 'unbiased')$cov[1, 1]),
  list(name = 'counts', values = values, weights = frequency, mean = weighted.mean(values, frequency),
    frequency = weighted_variance(values, frequency, 'frequency'), expanded_variance = var(rep(values, frequency)),
    reliability = weighted_variance(values, frequency, 'reliability'), population = weighted_variance(values, frequency, 'population'))
)

writeLines(json(list(reference = R.version.string, generator = 'tests/generate-distribution-fixtures.R',
  conventions = 'R d/p/q functions; logcdf/logsf use log.p=TRUE and lower.tail=FALSE; nonfinite numbers are strings',
  cases = cases)), 'tests/fixtures/distributions-r.json')
writeLines(json(list(reference = R.version.string, generator = 'tests/generate-distribution-fixtures.R',
  conventions = 'var/sd/cov use n - 1; quantile type 7; weighted variance definitions recorded per value',
  seed = 20261012, descriptive = descriptive, pairs = pairs, weighted = weighted)), 'tests/fixtures/descriptive-r.json')
cat(length(cases), 'distribution cases written\n')
