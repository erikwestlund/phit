# Run from the repository root: Rscript tests/generate-formula-fixtures.R
# Base R reference for model_matrix/fit_lm/fit_glm. Runtime tests read the saved JSON only.
# Character columns sort in the C locale, which the extension documents as its level order.
invisible(Sys.setlocale("LC_COLLATE", "C"))
RNGkind("Mersenne-Twister", "Inversion", "Rejection")
set.seed(20261015)

json <- function(x) {
  if (is.null(x)) return("null")
  if (is.list(x)) {
    values <- vapply(x, json, "")
    if (!is.null(names(x))) return(paste0("{", paste(paste0('"', names(x), '":', values), collapse = ","), "}"))
    return(paste0("[", paste(values, collapse = ","), "]"))
  }
  if (length(x) != 1) return(json(unname(as.list(x))))
  if (is.na(x)) return("null")
  if (is.logical(x)) return(if (x) "true" else "false")
  if (is.character(x)) return(paste0('"', gsub('"', '\\"', gsub("\\", "\\\\", x, fixed = TRUE), fixed = TRUE), '"'))
  if (!is.finite(x)) stop("nonfinite reference value")
  sprintf("%.17g", x)
}
rows <- function(m) unname(lapply(seq_len(nrow(m)), function(i) unname(as.list(m[i, ]))))
column <- function(v) {
  if (is.factor(v)) {
    codes <- as.integer(v) - 1L
    return(list(levels = as.list(levels(v)), codes = lapply(codes, function(c) if (is.na(c)) NULL else c),
      ordered = is.ordered(v)))
  }
  lapply(v, function(e) if (is.na(e)) NULL else e)
}
frame_json <- function(d) setNames(lapply(d, column), names(d))

# Repository datasets.
housing <- read.csv("docs/data/housing.csv")
schools <- read.csv("docs/data/schools.csv")
learning <- read.csv("docs/data/learning.csv")
clinic <- read.csv("docs/data/clinic.csv")
cbpp <- read.csv("docs/data/cbpp.csv", colClasses = c("character", "integer", "integer", "character"))
cbpp$herd <- factor(cbpp$herd, levels = as.character(1:15))
cbpp$period <- factor(cbpp$period, levels = as.character(1:4))

# Synthetic data: factors with a custom reference level and an unused level,
# ordered factors, logicals, characters, missing values, weights and exposure.
n <- 90
x1 <- round(rnorm(n), 3)
x2 <- round(runif(n, 1, 5), 3)
x2[c(4, 31, 77)] <- NA
f <- sample(c("north", "south", "east"), n, TRUE)
f[c(12, 50)] <- NA
g <- factor(sample(c("control", "low", "high"), n, TRUE), levels = c("control", "low", "high", "unused"))
o <- factor(sample(c("small", "medium", "large", "huge"), n, TRUE), levels = c("small", "medium", "large", "huge"), ordered = TRUE)
l <- runif(n) < 0.45
l[c(8, 66)] <- NA
w <- round(runif(n, 0.5, 3), 2)
expo <- round(runif(n, 0.5, 4), 2)
day <- as.numeric(as.Date("2024-01-01") + sample(0:400, n, TRUE))
fe <- c(north = 0.4, south = -0.3, east = 0)
ge <- c(control = 0, low = 0.5, high = 1.1, unused = 0)
eta <- 0.3 + 0.8 * x1 + ifelse(is.na(f), 0, fe[f]) + ge[as.character(g)] + 0.2 * as.integer(o)
y <- round(2 + 1.5 * x1 + ifelse(is.na(f), 0, fe[f]) + ge[as.character(g)] + 0.3 * as.integer(o) + rnorm(n, 0, 0.7), 3)
y[c(20, 45)] <- NA
bin <- rbinom(n, 1, plogis(eta - 0.8))
cnt <- rpois(n, expo * exp(0.2 + 0.4 * x1 + ifelse(is.na(f), 0, fe[f])))
ans <- factor(ifelse(rbinom(n, 1, plogis(eta - 0.5)) == 1, "yes", "no"), levels = c("no", "yes"))
trials <- sample(4:12, n, TRUE)
succ <- rbinom(n, trials, plogis(0.2 + 0.6 * x1))
synthetic <- data.frame(y = y, x1 = x1, x2 = x2, f = f, g = g, o = o, l = l, w = w, expo = expo, day = day,
  bin = bin, cnt = cnt, ans = ans, trials = trials, succ = succ, stringsAsFactors = FALSE)
# Columns for the dot formula only.
small <- synthetic[, c("y", "x1", "f", "l", "o")]

datasets <- list(housing = housing, schools = schools, learning = learning, clinic = clinic, cbpp = cbpp,
  synthetic = synthetic, small = small)

case <- function(name, dataset, fun, formula, family = NULL, link = NULL, weights = NULL)
  list(name = name, dataset = dataset, `function` = fun, formula = formula, family = family, link = link, weights = weights)
cases <- list(
  # model.matrix: names, values, assign, deletion and offsets.
  case("mm_housing_interaction", "housing", "model_matrix", "monthly_rent ~ area_m2 * bedrooms"),
  case("mm_housing_transforms", "housing", "model_matrix", "log(monthly_rent) ~ log(area_m2) + sqrt(bedrooms) + I(area_m2^2) + I(area_m2/bedrooms)"),
  case("mm_housing_poly_raw", "housing", "model_matrix", "monthly_rent ~ poly(area_m2, 3, raw = TRUE)"),
  case("mm_housing_poly_orthogonal", "housing", "model_matrix", "monthly_rent ~ poly(area_m2, 2)"),
  case("mm_housing_dot", "housing", "model_matrix", "monthly_rent ~ ."),
  case("mm_housing_factor_call", "housing", "model_matrix", "monthly_rent ~ factor(bedrooms) + area_m2"),
  case("mm_schools_character", "schools", "model_matrix", "score ~ study_hours + school"),
  case("mm_schools_slopes", "schools", "model_matrix", "score ~ study_hours:school + prior_score"),
  case("mm_cbpp_factors", "cbpp", "model_matrix", "incidence ~ period + herd"),
  case("mm_cbpp_binomial_matrix", "cbpp", "model_matrix", "cbind(incidence, size - incidence) ~ period"),
  case("mm_cbpp_offset", "cbpp", "model_matrix", "incidence ~ period + offset(log(size))"),
  case("mm_factor_numeric", "synthetic", "model_matrix", "y ~ f * x1"),
  case("mm_factor_descriptor", "synthetic", "model_matrix", "y ~ g + x1"),
  case("mm_ordered", "synthetic", "model_matrix", "y ~ o + x1"),
  case("mm_ordered_no_intercept", "synthetic", "model_matrix", "y ~ 0 + o"),
  case("mm_ordered_numeric", "synthetic", "model_matrix", "y ~ o:x1"),
  case("mm_logical", "synthetic", "model_matrix", "y ~ l + x1"),
  case("mm_logical_no_intercept", "synthetic", "model_matrix", "y ~ x1 + l - 1"),
  case("mm_factor_factor", "synthetic", "model_matrix", "y ~ f:g"),
  case("mm_factor_factor_main", "synthetic", "model_matrix", "y ~ f * g"),
  case("mm_factor_factor_no_intercept", "synthetic", "model_matrix", "y ~ f + g + 0"),
  case("mm_reverse_order", "synthetic", "model_matrix", "y ~ g:f + f"),
  case("mm_power", "synthetic", "model_matrix", "y ~ (x1 + f + l)^2"),
  case("mm_three_way", "synthetic", "model_matrix", "y ~ x1 * x2 * l"),
  case("mm_nesting", "synthetic", "model_matrix", "y ~ f/x1"),
  case("mm_removal", "synthetic", "model_matrix", "y ~ f * x1 - x1"),
  case("mm_intercept_restored", "synthetic", "model_matrix", "y ~ 0 + x1 + 1"),
  case("mm_one_sided", "synthetic", "model_matrix", "~ x2 + g"),
  case("mm_logic_transform", "synthetic", "model_matrix", "y ~ I(x1 > 0) + I((x1 + 1)^2) + log(expo, base = 2) + exp(x1) + abs(x1)"),
  case("mm_offsets", "synthetic", "model_matrix", "y ~ x1 + offset(log(expo)) + offset(x2)"),
  case("mm_dates", "synthetic", "model_matrix", "y ~ day + I(day - 19723)"),
  case("mm_dot_mixed", "small", "model_matrix", "y ~ ."),
  case("mm_factor_response", "synthetic", "model_matrix", "ans ~ x1"),
  # lm(): fit_ols contract plus names.
  case("lm_housing", "housing", "lm", "monthly_rent ~ area_m2"),
  case("lm_housing_log", "housing", "lm", "log(monthly_rent) ~ log(area_m2) + bedrooms"),
  case("lm_housing_poly", "housing", "lm", "monthly_rent ~ poly(area_m2, 2)"),
  case("lm_schools", "schools", "lm", "score ~ study_hours + prior_score + school"),
  case("lm_schools_interaction", "schools", "lm", "score ~ study_hours * prior_score"),
  case("lm_schools_no_intercept", "schools", "lm", "score ~ 0 + school + study_hours"),
  case("lm_missing_factor", "synthetic", "lm", "y ~ x1 + x2 + f"),
  case("lm_unused_level", "synthetic", "lm", "y ~ g + x1"),
  case("lm_ordered", "synthetic", "lm", "y ~ o + x1 + l"),
  case("lm_weighted_interaction", "synthetic", "lm", "y ~ f * x1", weights = "w"),
  case("lm_offset", "synthetic", "lm", "y ~ x1 + g + offset(x2)"),
  case("lm_through_origin", "synthetic", "lm", "y ~ x1 - 1"),
  case("lm_transforms", "synthetic", "lm", "y ~ poly(x1, 2, raw = TRUE) + I(x2^2) + log(expo)"),
  case("lm_intercept_only", "synthetic", "lm", "y ~ 1"),
  # glm(): binomial/probit/poisson/gaussian over the existing cores.
  case("glm_learning_logit", "learning", "glm", "passed ~ study_hours + practice_score", "binomial", "logit"),
  case("glm_learning_probit", "learning", "glm", "passed ~ study_hours * practice_score", "binomial", "probit"),
  case("glm_cbpp_cbind", "cbpp", "glm", "cbind(incidence, size - incidence) ~ period", "binomial", "logit"),
  case("glm_cbpp_proportion", "cbpp", "glm", "I(incidence/size) ~ period", "binomial", "logit", "size"),
  case("glm_cbpp_poisson_offset", "cbpp", "glm", "incidence ~ period + offset(log(size))", "poisson", "log"),
  case("glm_clinic_poisson", "clinic", "glm", "calls ~ log(visitors) + staff", "poisson", "log"),
  case("glm_binary_factor", "synthetic", "glm", "bin ~ x1 + f + g", "binomial", "logit"),
  case("glm_logical_response", "synthetic", "glm", "l ~ x1 + o", "binomial", "logit"),
  case("glm_factor_response", "synthetic", "glm", "ans ~ x1 + f", "binomial", "logit"),
  case("glm_factor_response_probit", "synthetic", "glm", "ans ~ x1 * l", "binomial", "probit"),
  case("glm_weighted_binomial", "synthetic", "glm", "I(succ/trials) ~ x1 + f", "binomial", "logit", "trials"),
  case("glm_poisson_offset", "synthetic", "glm", "cnt ~ x1 + f + offset(log(expo))", "poisson", "log"),
  case("glm_poisson_weighted", "synthetic", "glm", "cnt ~ x1 + g", "poisson", "log", "w"),
  case("glm_gaussian", "synthetic", "glm", "y ~ x1 + f + offset(x2)", "gaussian", "identity", "w"),
  case("glm_gaussian_no_intercept", "synthetic", "glm", "y ~ 0 + f + x1", "gaussian", "identity")
)

retained <- function(n, action) setdiff(seq_len(n), as.integer(action)) - 1L
fit_model <- function(spec, d) {
  f <- as.formula(spec$formula, env = globalenv())
  wexpr <- if (is.null(spec$weights)) NULL else as.name(spec$weights)
  if (spec[["function"]] == "lm") return(eval(bquote(lm(f, data = d, weights = .(wexpr)))))
  fam <- get(spec$family)(link = spec$link)
  fit <- eval(bquote(glm(f, family = fam, data = d, weights = .(wexpr), control = glm.control(epsilon = 1e-14, maxit = 100))))
  stopifnot(fit$converged)
  # glm.fit stops on relative deviance change, which fixes coefficients only to about
  # sqrt(epsilon). Restart from R's own solution until the coefficients stop moving.
  for (polish in 1:3) {
    start <- coef(fit)
    fit <- eval(bquote(glm(f, family = fam, data = d, weights = .(wexpr), start = start,
      control = glm.control(epsilon = 1e-16, maxit = 50))))
    if (max(abs(coef(fit) - start) / (1 + abs(start))) < 1e-13) break
  }
  stopifnot(max(abs(coef(fit) - start) / (1 + abs(start))) < 1e-12)
  fit
}

fixtures <- lapply(cases, function(spec) {
  d <- datasets[[spec$dataset]]
  n <- nrow(d)
  if (spec[["function"]] == "model_matrix") {
    f <- as.formula(spec$formula, env = globalenv())
    mf <- model.frame(f, d, na.action = na.omit)
    mm <- model.matrix(terms(mf), mf)
    y <- model.response(mf)
    expected <- list(columns = as.list(colnames(mm)), x = rows(mm), assign = as.list(attr(mm, "assign")),
      terms = as.list(attr(terms(mf), "term.labels")), intercept = attr(terms(mf), "intercept") == 1,
      retained_rows = as.list(retained(n, attr(mf, "na.action"))),
      dropped_rows = as.list(as.integer(attr(mf, "na.action")) - 1L))
    if (is.factor(y)) {
      expected$y <- as.list(as.integer(y) - 1)
      expected$response_levels <- as.list(levels(y))
    } else if (is.matrix(y)) {
      expected$y <- rows(y)
    } else if (!is.null(y)) {
      expected$y <- as.list(as.numeric(y))
    }
    if (!is.null(model.offset(mf))) expected$offset <- as.list(model.offset(mf))
    return(c(spec, list(expected = expected)))
  }
  fit <- fit_model(spec, d)
  stopifnot(!anyNA(coef(fit)))
  s <- summary(fit)
  p <- length(coef(fit))
  gaussian_like <- spec[["function"]] == "lm" || spec$family == "gaussian"
  ci <- if (gaussian_like) {
    q <- qt(0.975, df.residual(fit))
    cbind(coef(fit) - q * s$coefficients[, 2], coef(fit) + q * s$coefficients[, 2])
  } else confint.default(fit)
  expected <- list(columns = as.list(names(coef(fit))), coefficients = unname(as.list(coef(fit))),
    std_errors = unname(as.list(s$coefficients[, 2])), p_values = unname(as.list(s$coefficients[, 4])),
    conf_int = rows(ci), covariance = rows(vcov(fit)), df_residual = df.residual(fit),
    fitted = unname(as.list(fitted(fit))),
    log_likelihood = as.numeric(logLik(fit)),
    terms = as.list(attr(terms(fit), "term.labels")),
    retained_rows = as.list(retained(n, fit$na.action)))
  statistic <- if (gaussian_like) "t_statistics" else "z_statistics"
  expected[[statistic]] <- unname(as.list(s$coefficients[, 3]))
  if (spec[["function"]] == "lm") {
    expected$residuals <- unname(as.list(resid(fit)))
    expected$rss <- sum(weighted.residuals(fit)^2)
    expected$sigma <- s$sigma
    expected$r_squared <- s$r.squared
    expected$adj_r_squared <- s$adj.r.squared
  } else {
    expected$residuals <- unname(as.list(resid(fit, type = "response")))
    expected$deviance <- deviance(fit)
    expected$null_deviance <- fit$null.deviance
    expected$aic <- AIC(fit)
  }
  c(spec, list(expected = expected))
})

writeLines(json(list(reference = R.version.string, collation = "C",
  datasets = lapply(datasets, frame_json), cases = unname(fixtures))), "tests/fixtures/formula-r.json")
cat(length(fixtures), "formula reference cases written\n")
