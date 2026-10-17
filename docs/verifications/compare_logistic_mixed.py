"""Fresh PHP fits vs lme4 fixtures; independent SciPy marginal likelihood checks."""
from pathlib import Path
import hashlib,json,os,platform,subprocess,warnings
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq,minimize
from scipy.special import expit,gammaln,logsumexp
ROOT=Path(__file__).resolve().parents[2]
PHP=os.environ.get('PHP','/opt/homebrew/bin/php' if Path('/opt/homebrew/bin/php').exists() else 'php')
EXT=Path(os.environ.get('STATS_EXTENSION',str(ROOT/'target/release'/('libphit.dylib' if platform.system()=='Darwin' else 'libphit.so'))))
INFERENCE={'covariance','std_errors','z_statistics','p_values','conf_int'}
LIKELIHOOD={'log_likelihood','neg2_log_likelihood','aic','bic'}
def compare(a,b,key):
    if isinstance(b,list):
        assert len(a)==len(b),key
        for i,v in enumerate(b):compare(a[i],v,key+f'/{i}')
    elif isinstance(b,bool):assert a==b,key
    else:
        quantity=key.split('/')[1]
        absolute,relative=(2e-5,2e-3) if quantity in INFERENCE else ((1e-6,1e-8) if quantity in LIKELIHOOD else (2e-5,2e-4))
        assert np.isfinite(a) and abs(a-b)<=absolute+relative*abs(b),(key,a,b)

def evaluator(case):
    x=np.asarray(case['x']);y=np.asarray(case['y']);trials=np.asarray(case['weights']);offset=np.asarray(case['offset']);groups=np.asarray(case['groups'])
    members=[np.flatnonzero(groups==g) for g in dict.fromkeys(groups)]
    constant=np.sum(gammaln(trials+1)-gammaln(trials*y+1)-gammaln(trials*(1-y)+1))
    def evaluate(beta,tau,points=None):
        eta=offset+x@beta;ll=float(constant);max_relative_error=0.
        if points is not None:
            nodes,weights=np.polynomial.hermite_e.hermegauss(points);weights/=np.sqrt(2*np.pi)
        for rows in members:
            e=eta[rows];v=y[rows];w=trials[rows]
            def kernel(u):
                z=e+tau*u
                return -.5*u*u-np.sum(w*(v*np.logaddexp(0,-z)+(1-v)*np.logaddexp(0,z)))
            if tau==0:mode=0.
            else:
                bound=tau*w.sum()+1
                mode=brentq(lambda u:tau*np.sum(w*(v-expit(e+tau*u)))-u,-bound,bound,xtol=1e-13)
            precision=1+tau*tau*np.sum(w*expit(e+tau*mode)*expit(-e-tau*mode))
            if points is not None:
                ll+=logsumexp([np.log(weight)+kernel(mode+z/np.sqrt(precision))+.5*z*z for z,weight in zip(nodes,weights)])-.5*np.log(precision)
            else:
                # QUADPACK adaptive integration after centering/scaling at the
                # independent Brent posterior mode; no Hermite nodes used here.
                height=kernel(mode)
                integral,error=quad(lambda z:np.exp(kernel(mode+z/np.sqrt(precision))-height),
                    -np.inf,np.inf,epsabs=1e-11,epsrel=1e-11,limit=200)
                max_relative_error=max(max_relative_error,error/integral)
                ll+=height+np.log(integral)-.5*np.log(2*np.pi)-.5*np.log(precision)
        return float(ll),float(max_relative_error)
    return evaluate

def run(exact_fit=True):
    fixture_path=ROOT/'tests/fixtures/logistic-mixed-r.json';fixture=json.loads(fixture_path.read_text())
    results=[]
    for case in fixture['cases']:
        fit=json.loads(subprocess.run([PHP,'-n','-d',f'extension={EXT}',str(ROOT/'tests/logistic-mixed-fit.php')],input=json.dumps(case),capture_output=True,text=True,check=True).stdout)
        assert case['reference']['optimizer_code']==[0] and not case['reference']['warnings'],case['name']
        for key,value in case['expected'].items():compare(fit[key],value,case['name']+'/'+key)
        assert fit['n_parameters']==len(case['x'][0])+1 and fit['df_residual']==len(case['y'])-fit['n_parameters']
        independent=evaluator(case)
        same_approximation,_=independent(fit['coefficients'],fit['random_intercept_sd'],case['quadrature_points'])
        assert abs(same_approximation-fit['log_likelihood'])<2e-8,(case['name'],same_approximation,fit['log_likelihood'])
        integrated,error=independent(fit['coefficients'],fit['random_intercept_sd'])
        assert error<1e-9,(case['name'],error)
        results.append(dict(name=case['name'],dataset=case['dataset'],quadrature_points=case['quadrature_points'],
            R=case['expected'],R_reference=case['reference'],PHP=fit,
            scipy_same_approximation_log_likelihood=same_approximation,scipy_integrated_log_likelihood=integrated,
            integration_relative_error_estimate=error,approximation_log_likelihood_error=fit['log_likelihood']-integrated,
            max_coefficient_difference=max(abs(a-b) for a,b in zip(fit['coefficients'],case['expected']['coefficients']))))
    import statsmodels.api as sm
    import statsmodels,scipy
    boundary=next(c for c in fixture['cases'] if c['name']=='boundary_25')
    y=np.asarray(boundary['y']);w=np.asarray(boundary['weights']);x=np.asarray(boundary['x'])
    glm=sm.GLM(np.column_stack((y*w,(1-y)*w)),x,family=sm.families.Binomial(),offset=boundary['offset']).fit(tol=1e-12)
    native=next(r['PHP'] for r in results if r['name']=='boundary_25')
    assert np.allclose(glm.params,native['coefficients'],atol=1e-10,rtol=0)
    assert np.allclose(glm.cov_params(),native['covariance'],atol=1e-8,rtol=1e-6)
    assert abs(glm.llf-native['log_likelihood'])<1e-10
    exact=None
    if exact_fit:
        case=next(c for c in fixture['cases'] if c['name']=='cbpp_25');independent=evaluator(case)
        y=np.asarray(case['y']);w=np.asarray(case['weights']);x=np.asarray(case['x'])
        glm=sm.GLM(np.column_stack((y*w,(1-y)*w)),x,family=sm.families.Binomial()).fit()
        start=np.r_[glm.params,1.]
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            fitted=minimize(lambda t:-independent(t[:-1],t[-1])[0],start,method='L-BFGS-B',
                bounds=[(None,None)]*x.shape[1]+[(0.,20.)],options=dict(ftol=1e-13,gtol=1e-7,maxiter=500))
        assert fitted.success,(fitted.message,fitted.x)
        assert not captured,[str(w.message) for w in captured]
        native=next(r['PHP'] for r in results if r['name']=='cbpp_25')
        assert np.allclose(fitted.x[:-1],native['coefficients'],atol=5e-5,rtol=5e-5)
        assert abs(fitted.x[-1]-native['random_intercept_sd'])<5e-5
        exact=dict(dataset='cbpp',coefficients=fitted.x[:-1].tolist(),random_intercept_sd=float(fitted.x[-1]),
            log_likelihood=float(-fitted.fun),converged=bool(fitted.success),evaluations=fitted.nfev,
            message=str(fitted.message),method='SciPy L-BFGS-B + QUADPACK integration',start='statsmodels binomial GLM, SD=1')
    report=dict(versions=dict(R=fixture['r'],lme4=fixture['lme4'],PHP=subprocess.check_output([PHP,'-n','-r','echo PHP_VERSION;'],text=True),
        python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,statsmodels=statsmodels.__version__),
        tolerances=dict(inference=dict(absolute=2e-5,relative=2e-3),estimates=dict(absolute=2e-5,relative=2e-4),
            likelihood=dict(absolute=1e-6,relative=1e-8),independent_likelihood_absolute=2e-8),
        extension_sha256=hashlib.sha256(EXT.read_bytes()).hexdigest(),fixture_sha256=hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        data_sha256=hashlib.sha256((ROOT/'docs/data/cbpp.csv').read_bytes()).hexdigest(),
        results=results,exact_fit=exact,statsmodels_boundary_check=True)
    (ROOT/'docs/verifications/logistic-mixed-results.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Passed 28 R/PHP comparisons, independent likelihood/integration checks, and statsmodels zero-variance reduction.')
    if exact:print('Independent cbpp marginal ML fit:',exact)
    return report
if __name__=='__main__':run()
