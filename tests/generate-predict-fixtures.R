# Run from the repository root: Rscript tests/generate-predict-fixtures.R
# predict() references for predict. Runtime tests read the saved JSON only.
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
column <- function(v) {
  if (is.factor(v)) {
    codes <- as.integer(v) - 1L
    return(list(levels = as.list(levels(v)), codes = lapply(codes, function(c) if (is.na(c)) NULL else c),
      ordered = is.ordered(v)))
  }
  lapply(v, function(e) if (is.na(e)) NULL else e)
}
frame <- function(d) lapply(d, column)
values <- function(v) lapply(unname(v), function(e) if (is.na(e)) NULL else e)

cars <- mtcars
cars$name <- NULL
cars$fast <- cars$qsec < 17
cars$gear_label <- c("three", "four", "five")[match(cars$gear, c(3, 4, 5))]
# Out-of-range values exercise poly() and factor levels absent from a small newdata.
cars_new <- data.frame(
  mpg = c(30, 21, 15.8, 10.4, NA, 19.2), hp = c(52, 110, 335, 400, NA, 150), wt = c(1.5, 2.6, 3.57, 5.6, 3, 3.2),
  am = c(1, 0, 1, 0, 1, 0), cyl = c(4, 6, 8, 8, 4, 6), qsec = c(19, 16.5, 14.6, 18, 17, 20),
  disp = c(80, 160, 351, 460, 120, 200), vs = c(1, 0, 0, 0, 1, 1), gear = c(4, 3, 5, 3, 4, 4))
cars_new$fast <- cars_new$qsec < 17
cars_new$gear_label <- c("three", "four", "five")[match(cars_new$gear, c(3, 4, 5))]

breaks <- warpbreaks
breaks$tension <- factor(breaks$tension, levels = c("L", "M", "H"), ordered = TRUE)
breaks$hours <- round(runif(nrow(breaks), 2, 8), 2)
breaks_new <- data.frame(
  wool = factor(c("B", "A", "B", NA), levels = c("A", "B")),
  tension = factor(c("H", "L", "M", "L"), levels = c("L", "M", "H"), ordered = TRUE),
  hours = c(1, 4, 12, 5), breaks = NA_real_)

cbpp <- read.csv("docs/data/cbpp.csv", colClasses = c("character", "integer", "integer", "character"))
cbpp$herd <- factor(cbpp$herd, levels = as.character(1:15))
cbpp$period <- factor(cbpp$period, levels = as.character(1:4))
cbpp_new <- data.frame(herd = factor(c("1", "7", "15", "16"), levels = as.character(1:16)),
  period = factor(c("1", "2", "4", "3"), levels = as.character(1:4)), size = c(10, 20, 5, 8), incidence = NA_integer_)

schools <- read.csv("docs/data/schools.csv")
schools_new <- schools[c(1, 50, 120), setdiff(names(schools), "score")]
schools_new <- rbind(schools_new, transform(schools_new[1, ], school = "unseen school"))
schools_new$study_hours[2] <- NA

subjects <- sprintf("s%02d", 1:24)
items <- sprintf("i%d", 1:8)
panel <- expand.grid(rep = 1:2, item = items, subject = subjects, stringsAsFactors = FALSE)
n <- nrow(panel)
u_subject <- rnorm(24, 0, 1.2); u_slope <- rnorm(24, 0, 0.4); u_item <- rnorm(8, 0, 0.8)
si <- match(panel$subject, subjects); ii <- match(panel$item, items)
panel$x <- round(rnorm(n), 3)
panel$y <- round(3 + 0.9 * panel$x + u_subject[si] + u_slope[si] * panel$x + u_item[ii] + rnorm(n, 0, 0.9), 3)
panel$yb <- rbinom(n, 1, plogis(-0.4 + 0.8 * panel$x + u_subject[si]))
panel$rep <- NULL
panel_new <- data.frame(subject = c("s01", "s07", "s24", "s99", "s03"), item = c("i1", "i8", "i9", "i2", "i4"),
  x = c(-1.2, 0, 2.5, 0.3, NA))

datasets <- list(cars = cars, breaks = breaks, cbpp = cbpp, schools = schools, panel = panel)
newdata <- list(cars = cars_new, breaks = breaks_new, cbpp = cbpp_new, schools = schools_new, panel = panel_new)

prediction <- function(p) {
  if (is.list(p) && !is.null(p$fit)) {
    out <- list()
    if (is.matrix(p$fit)) {
      out$fit <- values(p$fit[, "fit"]); out$lower <- values(p$fit[, "lwr"]); out$upper <- values(p$fit[, "upr"])
    } else out$fit <- values(p$fit)
    out$se_fit <- values(p$se.fit)
    if (!is.null(p$df)) out$df <- p$df
    if (!is.null(p$residual.scale)) out$residual_scale <- p$residual.scale
    return(out)
  }
  if (is.matrix(p)) return(list(fit = values(p[, "fit"]), lower = values(p[, "lwr"]), upper = values(p[, "upr"])))
  list(fit = values(p))
}

cases <- list()
add <- function(name, dataset, fun, formula, options, expected, args = list())
  cases[[length(cases) + 1]] <<- list(name = name, dataset = dataset, `function` = fun, formula = formula,
    args = args, options = options, expected = expected)

lm_formula <- "mpg ~ poly(hp, 3) + factor(cyl) + wt:am + fast + gear_label"
fit <- lm(as.formula(lm_formula), cars)
add("lm_poly_factor_confidence", "cars", "lm", lm_formula, list(se_fit = TRUE, interval = "confidence"),
  prediction(predict(fit, cars_new, se.fit = TRUE, interval = "confidence")))
add("lm_poly_factor_prediction_90", "cars", "lm", lm_formula, list(interval = "prediction", level = 0.9),
  prediction(predict(fit, cars_new, interval = "prediction", level = 0.9)))
fit <- lm(mpg ~ log(disp) * wt + I(hp > 150) - 1, cars)
add("lm_no_intercept_interaction", "cars", "lm", "mpg ~ log(disp) * wt + I(hp > 150) - 1", list(se_fit = TRUE),
  prediction(predict(fit, cars_new, se.fit = TRUE)))
fit <- lm(breaks ~ wool * tension + offset(log(hours)), breaks)
add("lm_ordered_interaction_offset", "breaks", "lm", "breaks ~ wool * tension + offset(log(hours))",
  list(se_fit = TRUE, interval = "confidence"), prediction(predict(fit, breaks_new, se.fit = TRUE, interval = "confidence")))
fit <- lm(mpg ~ wt + hp, cars, weights = disp)
add("lm_weighted_confidence", "cars", "lm", "mpg ~ wt + hp", list(se_fit = TRUE, interval = "confidence"),
  prediction(predict(fit, cars_new, se.fit = TRUE, interval = "confidence")), list(weights = "disp"))

# R's vcov(glm) reuses the weights of the last IRLS iteration, computed one step
# before the final coefficients; phit evaluates the information at the final
# coefficients. se_fit_final is that standard error, recomputed here in R; R's own
# predict() value stays as se_fit for a looser check.
final_se <- function(fit, nd, type) {
  tt <- delete.response(terms(fit))
  X <- model.matrix(fit)
  eta <- drop(X %*% coef(fit)) + if (is.null(fit$offset)) 0 else fit$offset
  w <- fit$prior.weights * fit$family$mu.eta(eta)^2 / fit$family$variance(fit$family$linkinv(eta))
  V <- summary(fit)$dispersion * solve(crossprod(X * sqrt(w)))
  mf <- model.frame(tt, nd, xlev = fit$xlevels, na.action = na.pass)
  x <- model.matrix(tt, mf)
  se <- sqrt(rowSums((x %*% V) * x))
  if (type == "response") {
    eta_new <- predict(fit, nd, type = "link")
    se <- se * abs(fit$family$mu.eta(eta_new))
  }
  values(se)
}
glm_case <- function(name, dataset, formula, family, link, fit) {
  nd <- newdata[[dataset]]
  # glm()'s default deviance tolerance (1e-8) can leave probit coefficients ~1e-5
  # from the optimum; references use a converged fit, as phit iterates further.
  fit <- update(fit, control = glm.control(epsilon = 1e-14, maxit = 200))
  for (type in c("link", "response")) {
    expected <- prediction(predict(fit, nd, type = type, se.fit = TRUE))
    expected$se_fit_final <- final_se(fit, nd, type)
    add(paste0(name, "_", type), dataset, "glm", formula, list(type = type, se_fit = TRUE),
      expected, list(family = family, link = link))
  }
}
glm_case("glm_logit", "cars", "am ~ wt + poly(hp, 2)", "binomial", "logit",
  glm(am ~ wt + poly(hp, 2), binomial, cars))
glm_case("glm_probit", "cars", "vs ~ mpg + gear_label", "binomial", "probit",
  glm(vs ~ mpg + gear_label, binomial("probit"), cars))
glm_case("glm_poisson_offset", "breaks", "breaks ~ wool + tension + offset(log(hours))", "poisson", "log",
  glm(breaks ~ wool + tension + offset(log(hours)), poisson, breaks))
glm_case("glm_grouped_binomial", "cbpp", "cbind(incidence, size - incidence) ~ period", "binomial", "logit",
  glm(cbind(incidence, size - incidence) ~ period, binomial, cbpp))
glm_case("glm_gaussian", "cars", "mpg ~ wt + factor(cyl)", "gaussian", "identity",
  glm(mpg ~ wt + factor(cyl), gaussian, cars))

control <- lmerControl(optimizer = "bobyqa", optCtrl = list(rhoend = 1e-9, maxfun = 100000), boundary.tol = 1e-3)
mixed_case <- function(name, dataset, fun, formula, fit, args = list()) {
  nd <- newdata[[dataset]]
  add(paste0(name, "_conditional"), dataset, fun, formula, list(allow_new_levels = TRUE),
    prediction(predict(fit, nd, allow.new.levels = TRUE)), args)
  add(paste0(name, "_population"), dataset, fun, formula, list(random = FALSE),
    prediction(predict(fit, nd, re.form = NA)), args)
  if (fun == "glmer")
    add(paste0(name, "_response"), dataset, fun, formula, list(type = "response", allow_new_levels = TRUE),
      prediction(predict(fit, nd, type = "response", allow.new.levels = TRUE)), args)
}
mixed_case("lmer_schools", "schools", "lmer", "score ~ study_hours + (1 | school)",
  lmer(score ~ study_hours + (1 | school), schools, control = control))
mixed_case("lmer_panel_slopes_crossed", "panel", "lmer", "y ~ x + (x | subject) + (1 | item)",
  lmer(y ~ x + (x | subject) + (1 | item), panel, REML = FALSE, control = control), list(reml = FALSE))
mixed_case("glmer_cbpp", "cbpp", "glmer", "cbind(incidence, size - incidence) ~ period + (1 | herd)",
  glmer(cbind(incidence, size - incidence) ~ period + (1 | herd), cbpp, binomial,
    control = glmerControl(optimizer = "bobyqa", tolPwrss = 1e-12, optCtrl = list(rhoend = 1e-9, maxfun = 100000))))
mixed_case("glmer_panel_binary", "panel", "glmer", "yb ~ x + (1 | subject)",
  glmer(yb ~ x + (1 | subject), panel, binomial,
    control = glmerControl(optimizer = "bobyqa", tolPwrss = 1e-12, optCtrl = list(rhoend = 1e-9, maxfun = 100000))))

out <- list(
  generator = "tests/generate-predict-fixtures.R",
  r_version = R.version.string,
  lme4_version = as.character(packageVersion("lme4")),
  datasets = lapply(datasets, frame),
  newdata = lapply(newdata, frame),
  cases = cases)
writeLines(json(out), "tests/fixtures/predict-r.json")
cat(length(cases), "predict() references written\n")
