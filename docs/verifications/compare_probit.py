"""Estimate shared binary probit models in R, statsmodels and the PHP extension."""
import csv
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import numpy as np
import statsmodels
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def cases():
    ordinary = json.loads((ROOT / 'tests/fixtures/probit-r.json').read_text())['cases']
    first = ordinary[0]
    x = [row.copy() for row in first['x']]
    y = first['y'].copy()
    x[1][1], y[4] = None, None
    return [*ordinary, {'name': 'listwise_missing', 'x': x, 'y': y}]


def compare_case(case):
    design, outcomes = np.asarray(case['x'], dtype=float), np.asarray(case['y'], dtype=float)
    keep = ~(np.isnan(design).any(axis=1) | np.isnan(outcomes))
    x, y = design[keep], outcomes[keep]
    intercept = bool(np.any(np.all(x == 1, axis=0)))
    fit = sm.GLM(y,x,family=sm.families.Binomial(link=sm.families.links.Probit()),missing='raise').fit(maxiter=100,tol=1e-11,tol_criterion="params")
    assert fit.converged
    null_ll = np.sum(y * np.log(y.mean()) + (1-y) * np.log1p(-y.mean())) if intercept else -len(y)*np.log(2)
    python_result = dict(coefficients=fit.params.tolist(), covariance=fit.cov_params().tolist(),
                         std_errors=fit.bse.tolist(), z_statistics=fit.tvalues.tolist(),
                         p_values=fit.pvalues.tolist(), conf_int=fit.conf_int().tolist(),
                         fitted=fit.predict().tolist(), residuals=(y-fit.predict()).tolist(),
                         log_likelihood=float(fit.llf), deviance=float(-2*fit.llf),
                         null_deviance=float(-2*null_ll), aic=float(fit.aic), df_residual=int(fit.df_resid))
    suffix = '.dylib' if sys.platform == 'darwin' else '.so'
    extension = Path(os.environ.get('STATS_EXTENSION', ROOT / f'target/debug/libphp_stats{suffix}')).resolve()
    native_run = subprocess.run([os.environ.get('PHP','php'),'-n','-d',f'extension={extension}',str(HERE/'probit.php')],
                                input=json.dumps({'x':case['x'],'y':case['y']}),text=True,capture_output=True,check=True)
    native = json.loads(native_run.stdout)
    with tempfile.TemporaryDirectory() as directory:
        data = Path(directory)/'data.csv'
        with data.open('w',newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['y',*[f'x{j}' for j in range(design.shape[1])]])
            for row, outcome in zip(case['x'],case['y']):
                writer.writerow(['NA' if v is None else v for v in [outcome,*row]])
        run = subprocess.run([os.environ.get('RSCRIPT','Rscript'),str(HERE/'probit.R'),str(data)],
                             text=True,capture_output=True,check=True)
        reference = json.loads(run.stdout)
    errors = {}
    for key, expected in reference['result'].items():
        errors[key] = {}
        for name, result in [('statsmodels',python_result),('php-stats',native['result'])]:
            np.testing.assert_allclose(result[key],expected,atol=1e-9,rtol=1e-7,err_msg=f'{case["name"]}/{name}/{key}')
            errors[key][name] = float(np.max(np.abs(np.asarray(result[key])-expected)))
    retained, dropped = np.flatnonzero(keep).tolist(), np.flatnonzero(~keep).tolist()
    assert native['result']['retained_rows'] == retained and native['result']['dropped_rows'] == dropped
    assert native['result']['n_observations'] == reference['n_observations'] == len(retained)
    assert native['result']['n_input'] == len(outcomes) and native['result']['n_dropped'] == len(dropped)
    assert native['result']['converged'] and len(native['warnings']) == int(bool(dropped))
    if dropped:
        assert native['warnings'][0]['severity'] == 2
    return dict(name=case['name'], input={'x':case['x'],'y':case['y']},
                versions={'R':reference['version'],'statsmodels':statsmodels.__version__,'numpy':np.__version__,
                          'Python':platform.python_version(),'PHP':native['version']},
                **{'R':reference['result'],'statsmodels':python_result,'php-stats':native['result']},
                warnings=native['warnings'],max_absolute_errors_vs_R=errors)


if __name__ == '__main__':
    results = [compare_case(case) for case in cases()]
    (HERE/'probit-results.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    for result in results:
        print(f'{result["name"]}: R, statsmodels and PHP agree; n={result["php-stats"]["n_observations"]}')
