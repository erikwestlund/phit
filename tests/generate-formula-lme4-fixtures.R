# Run from the repository root: Rscript tests/generate-formula-lme4-fixtures.R
# lme4 references for stats_lmer/stats_glmer formulas. Runtime tests read the saved JSON only.
suppressPackageStartupMessages(library(lme4))
invisible(Sys.setlocale("LC_COLLATE", "C"))
RNGkind("Mersenne-Twister", "Inversion", "Rejection")
set.seed(20261016)

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
  if (is.character(x)) return(paste0('"', gsub('"', '\\"', x, fixed = TRUE), '"'))
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

schools <- read.csv("docs/data/schools.csv")
cbpp <- read.csv("docs/data/cbpp.csv", colClasses = c("character", "integer", "integer", "character"))
cbpp$herd <- factor(cbpp$herd, levels = as.character(1:15))
cbpp$period <- factor(cbpp$period, levels = as.character(1:4))

# Crossed/nested panel: 24 subjects x 8 items x 2 replicates, with missing values.
subjects <- sprintf("s%02d", 1:24)
items <- sprintf("i%d", 1:8)
panel <- expand.grid(rep = 1:2, item = items, subject = subjects, stringsAsFactors = FALSE)
n <- nrow(panel)
u_subject <- rnorm(24, 0, 1.2); u_slope <- rnorm(24, 0, 0.4); u_item <- rnorm(8, 0, 0.8)
u_cell <- rnorm(24 * 8, 0, 0.5)
si <- match(panel$subject, subjects); ii <- match(panel$item, items)
panel$x <- round(rnorm(n), 3)
panel$y <- round(3 + 0.9 * panel$x + u_subject[si] + u_slope[si] * panel$x + u_item[ii] +
  u_cell[(si - 1) * 8 + ii] + rnorm(n, 0, 0.9), 3)
panel$yb <- rbinom(n, 1, plogis(-0.4 + 0.8 * panel$x + u_subject[si]))
panel$y[c(5, 77, 300)] <- NA
panel$x[c(40, 211)] <- NA
panel$rep <- NULL
datasets <- list(schools = schools, cbpp = cbpp, panel = panel)

bar <- function(group, columns, correlated = TRUE) list(group = group, columns = columns, correlated = correlated)
case <- function(name, dataset, fun, formula, bars, reml = TRUE, nagq = 1)
  list(name = name, dataset = dataset, `function` = fun, formula = formula, bars = bars, reml = reml, quadrature_points = nagq)
cases <- list(
  case("lmer_schools_intercept_reml", "schools", "lmer", "score ~ study_hours + (1 | school)", list(bar("school", "(Intercept)"))),
  case("lmer_schools_intercept_ml", "schools", "lmer", "score ~ study_hours + prior_score + (1 | school)", list(bar("school", "(Intercept)")), reml = FALSE),
  case("lmer_panel_slopes", "panel", "lmer", "y ~ x + (x | subject)", list(bar("subject", c("(Intercept)", "x")))),
  case("lmer_panel_uncorrelated", "panel", "lmer", "y ~ x + (x || subject) + (1 | item)",
    list(bar("subject", c("(Intercept)", "x"), FALSE), bar("item", "(Intercept)"))),
  case("lmer_panel_crossed", "panel", "lmer", "y ~ x + (1 | subject) + (1 | item)",
    list(bar("subject", "(Intercept)"), bar("item", "(Intercept)"))),
  case("lmer_panel_slopes_crossed_ml", "panel", "lmer", "y ~ x + (x | subject) + (1 | item)",
    list(bar("subject", c("(Intercept)", "x")), bar("item", "(Intercept)")), reml = FALSE),
  case("lmer_panel_nested", "panel", "lmer", "y ~ x + (1 | subject/item)",
    list(bar("subject", "(Intercept)"), bar("item:subject", "(Intercept)"))),
  case("glmer_cbpp_laplace", "cbpp", "glmer", "cbind(incidence, size - incidence) ~ period + (1 | herd)", list(bar("herd", "(Intercept)"))),
  case("glmer_cbpp_agq", "cbpp", "glmer", "cbind(incidence, size - incidence) ~ period + (1 | herd)", list(bar("herd", "(Intercept)")), nagq = 9),
  case("glmer_panel_binary", "panel", "glmer", "yb ~ x + (1 | subject)", list(bar("subject", "(Intercept)")))
)

fixtures <- lapply(cases, function(spec) {
  d <- datasets[[spec$dataset]]
  f <- as.formula(spec$formula, env = globalenv())
  if (spec[["function"]] == "lmer") {
    fit <- lmer(f, data = d, REML = spec$reml,
      control = lmerControl(optimizer = "bobyqa", optCtrl = list(rhoend = 1e-9, maxfun = 100000), boundary.tol = 1e-3))
  } else {
    fit <- glmer(f, data = d, family = binomial("logit"), nAGQ = spec$quadrature_points,
      control = glmerControl(optimizer = "bobyqa", tolPwrss = 1e-12, optCtrl = list(rhoend = 1e-9, maxfun = 100000)))
  }
  beta <- fixef(fit); cv <- as.matrix(vcov(fit))
  expected <- list(columns = as.list(names(beta)), coefficients = unname(as.list(beta)), covariance = rows(cv),
    std_errors = unname(as.list(sqrt(diag(cv)))), n_observations = nobs(fit))
  vc <- VarCorr(fit)
  groups <- sub("\\.[0-9]+$", "", names(vc))
  covariances <- lapply(spec$bars, function(b) {
    k <- length(b$columns); out <- matrix(0, k, k)
    idx <- which(groups == b$group)
    if (b$correlated) {
      hit <- idx[vapply(vc[idx], function(v) identical(rownames(v), b$columns), FALSE)]
      stopifnot(length(hit) == 1)
      out <- unname(as.matrix(vc[[hit]]))
    } else for (j in seq_len(k)) {
      hit <- idx[vapply(vc[idx], function(v) identical(rownames(v), b$columns[j]), FALSE)]
      stopifnot(length(hit) == 1)
      out[j, j] <- vc[[hit]][1, 1]
    }
    rows(out)
  })
  # ranef(condVar = TRUE) postVar per bar: one k x k block per level (lme4's
  # level order). (x || g) splits into separate lme4 terms, so lme4 reports
  # only each coefficient's own variance: record the diagonal.
  re <- ranef(fit, condVar = TRUE)
  expected$random_cond_var <- lapply(spec$bars, function(b) {
    r <- re[[b$group]]; pv <- attr(r, "postVar"); levels <- as.list(rownames(r))
    if (is.list(pv)) {
      stopifnot(!b$correlated, identical(names(pv), b$columns))
      return(list(levels = levels, cond_var_diagonal = lapply(seq_len(nrow(r)), function(i)
        lapply(b$columns, function(column) pv[[column]][1, 1, i]))))
    }
    stopifnot(b$correlated, identical(colnames(r), b$columns))
    list(levels = levels, cond_var = lapply(seq_len(nrow(r)), function(i) rows(matrix(pv[, , i], length(b$columns)))))
  })
  if (spec[["function"]] == "lmer") {
    expected$random_covariances <- covariances
    expected$sigma <- sigma(fit)
    expected$residual_variance <- sigma(fit)^2
    expected$log_likelihood <- as.numeric(logLik(fit))
    expected$aic <- AIC(fit); expected$bic <- BIC(fit)
  } else {
    # lme4 nAGQ>1 drops the saturated binomial constant; restore it (as in logistic-mixed-r).
    mf <- model.frame(fit); y <- model.response(mf)
    if (is.matrix(y)) { m <- rowSums(y); y <- y[, 1] / m } else m <- rep(1, length(y))
    correction <- if (spec$quadrature_points > 1) sum(dbinom(round(y * m), m, y, log = TRUE)) else 0
    ll <- as.numeric(logLik(fit)) + correction; np <- length(beta) + 1
    expected$random_intercept_variance <- covariances[[1]][[1]][[1]]
    expected$log_likelihood <- ll
    expected$aic <- 2 * np - 2 * ll; expected$bic <- log(nobs(fit)) * np - 2 * ll
  }
  c(spec, list(expected = expected, reference = list(optimizer = "bobyqa", rhoend = 1e-9,
    optimizer_code = as.list(fit@optinfo$conv$opt), messages = as.list(unlist(fit@optinfo$conv$lme4$messages)))))
})

writeLines(json(list(reference = R.version.string, lme4 = as.character(packageVersion("lme4")), collation = "C",
  datasets = lapply(datasets, function(d) setNames(lapply(d, column), names(d))), cases = unname(fixtures))),
  "tests/fixtures/formula-lme4-r.json")
cat(length(fixtures), "lme4 formula reference cases written\n")
