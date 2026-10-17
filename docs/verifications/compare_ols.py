"""Re-estimate the shared fixture in R, statsmodels and the PHP extension."""
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


def statsmodels_result(x, y):
    design = np.asarray(x, dtype=float)
    outcomes = np.asarray(y, dtype=float)
    # All examples provide their intercept explicitly, just like PHP.
    retained = ~(np.isnan(design).any(axis=1) | np.isnan(outcomes))
    has_intercept = bool(np.any(np.all(design[retained] == 1, axis=0)))
    fit = sm.OLS(outcomes, design, missing='drop', hasconst=has_intercept).fit(use_t=True)
    return {
        'coefficients': fit.params.tolist(), 'covariance': fit.cov_params().tolist(),
        'std_errors': fit.bse.tolist(), 't_statistics': fit.tvalues.tolist(),
        'p_values': fit.pvalues.tolist(), 'conf_int': fit.conf_int(alpha=0.05).tolist(),
        'fitted': fit.fittedvalues.tolist(), 'residuals': fit.resid.tolist(),
        'rss': float(fit.ssr), 'residual_variance': float(fit.scale),
        'sigma': float(np.sqrt(fit.scale)), 'r_squared': float(fit.rsquared),
        'adj_r_squared': float(fit.rsquared_adj), 'df_residual': int(fit.df_resid),
        'log_likelihood': float(fit.llf),
    }


def compare_case(case):
    php = os.environ.get('PHP', 'php')
    suffix = '.dylib' if sys.platform == 'darwin' else '.so'
    extension = Path(os.environ.get('STATS_EXTENSION', ROOT / 'target/debug' / ('libphit' + suffix))).resolve()
    if not extension.is_file():
        raise FileNotFoundError(f'Build the extension first: {extension}')
    php_run = subprocess.run(
        [php, '-n', '-d', f'extension={extension}', str(HERE / 'ols.php')],
        input=json.dumps({'x': case['x'], 'y': case['y']}), text=True, capture_output=True, check=True,
    )
    native = json.loads(php_run.stdout)
    with tempfile.TemporaryDirectory() as directory:
        data = Path(directory) / 'data.csv'
        with data.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['y'] + [f'x{i}' for i in range(len(case['x'][0]))])
            for row, outcome in zip(case['x'], case['y']):
                writer.writerow(['NA' if v is None else v for v in [outcome, *row]])
        r_run = subprocess.run([os.environ.get('RSCRIPT', 'Rscript'), str(HERE / 'ols.R'), str(data)],
                               text=True, capture_output=True, check=True)
        reference = json.loads(r_run.stdout)
    python_result = statsmodels_result(case['x'], case['y'])
    errors = {}
    for key, expected in reference['result'].items():
        errors[key] = {}
        for name, result in [('statsmodels', python_result), ('phit', native['result'])]:
            actual = np.asarray(result[key])
            wanted = np.asarray(expected)
            np.testing.assert_allclose(actual, wanted, atol=1e-9, rtol=1e-7, err_msg=f'{case["name"]}/{name}/{key}')
            errors[key][name] = float(np.max(np.abs(actual - wanted)))
    retained = [i for i, (row, outcome) in enumerate(zip(case['x'], case['y']))
                if outcome is not None and all(v is not None for v in row)]
    dropped = [i for i in range(len(case['y'])) if i not in retained]
    assert native['result']['retained_rows'] == retained
    assert native['result']['dropped_rows'] == dropped
    assert native['result']['n_observations'] == reference['n_observations'] == len(retained)
    assert native['result']['n_input'] == len(case['y'])
    assert native['result']['n_dropped'] == len(dropped)
    assert len(native['warnings']) == int(bool(dropped))
    if dropped:
        assert native['warnings'][0]['severity'] == 2  # E_WARNING
    return {'name': case['name'], 'input': {'x': case['x'], 'y': case['y']},
            'versions': {'R': reference['version'], 'statsmodels': statsmodels.__version__,
                         'numpy': np.__version__, 'Python': platform.python_version(), 'PHP': native['version']},
            'R': reference['result'], 'statsmodels': python_result, 'phit': native['result'],
            'warnings': native['warnings'], 'max_absolute_errors_vs_R': errors}


def cases():
    fixtures = json.loads((ROOT / 'tests/fixtures/ols-r.json').read_text())['cases']
    ordinary = [case for case in fixtures if case['name'] not in ('perfect', 'zero')]
    missing = {'name': 'listwise_missing', 'x': [[1, 1], [1, None], [1, 2], [1, 3], [1, 4], [1, 5]],
               'y': [2.1, 99, 4.2, None, 8.1, 10.2]}
    return [*ordinary, missing]


if __name__ == '__main__':
    results = [compare_case(case) for case in cases()]
    output = HERE / 'ols-results.json'
    output.write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')
    for result in results:
        print(f'{result["name"]}: R, statsmodels and PHP agree; n={result["phit"]["n_observations"]}')
    print(f'Wrote {output.relative_to(ROOT)}')
