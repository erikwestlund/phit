"""One statsmodels benchmark worker; process startup is timed by the driver."""
import json
import os
import sys
import time

import numpy as np
import statsmodels
import statsmodels.api as sm

start = time.perf_counter()
data = np.loadtxt(sys.argv[1], delimiter=",", skiprows=1)
y, x = data[:, 0].copy(), data[:, 1:].copy()
del data
load_seconds = time.perf_counter() - start
indices = np.unique(np.linspace(0, len(y) - 1, 10, dtype=int))


def estimate():
    fit = sm.OLS(y, x, hasconst=True, missing="raise").fit(use_t=True)
    # Force lazy inference and both full observation-length outputs within timing.
    fitted, residuals = fit.fittedvalues, fit.resid
    return dict(coefficients=fit.params.tolist(), covariance=fit.cov_params().tolist(),
                std_errors=fit.bse.tolist(), t_statistics=fit.tvalues.tolist(),
                p_values=fit.pvalues.tolist(), conf_int=fit.conf_int().tolist(),
                rss=float(fit.ssr), residual_variance=float(fit.scale),
                sigma=float(np.sqrt(fit.scale)), r_squared=float(fit.rsquared),
                adj_r_squared=float(fit.rsquared_adj), df_residual=int(fit.df_resid),
                log_likelihood=float(fit.llf), fitted_sample=fitted[indices].tolist(),
                residuals_sample=residuals[indices].tolist())


estimate()  # One untimed warm-up.
times = []
for _ in range(int(sys.argv[2])):
    start = time.perf_counter()
    result = estimate()
    times.append(time.perf_counter() - start)
print(json.dumps(dict(version=statsmodels.__version__, numpy=np.__version__,
                      load_seconds=load_seconds, fit_seconds=times, result=result,
                      sample_indices=indices.tolist()), allow_nan=False))
