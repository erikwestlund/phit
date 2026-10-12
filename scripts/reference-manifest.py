#!/usr/bin/env python3
"""Index saved reference inputs/options, versions, quantities and test policies; no R needed."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
GENERATORS={'ols-r':'tests/generate-fixtures.R','logistic-r':'tests/generate-logistic-fixtures.R',
 'probit-r':'tests/generate-probit-poisson-fixtures.R','poisson-r':'tests/generate-probit-poisson-fixtures.R',
 'r-compatibility':'tests/generate-r-compatibility-fixtures.R','lmm-r':'tests/generate-lmm-fixtures.R',
 'lmm-public-r':'tests/generate-lmm-public-fixtures.py','lmm-expanded-r':'tests/generate-lmm-expanded-fixtures.R',
 'logistic-mixed-r':'tests/generate-logistic-mixed-fixtures.R','stress-r':'tests/generate-stress-fixtures.R'}
POLICY={'ols-r':'1e-9 + 1e-7*abs(reference); zero/perfect fit conventions tested separately',
 'logistic-r':'1e-9 + 1e-7*abs(reference)','probit-r':'1e-9 + 1e-7*abs(reference)','poisson-r':'1e-9 + 1e-7*abs(reference)',
 'r-compatibility':'1e-7 + 1e-6*abs(reference)',
 'lmm-r':'1e-7 + 1e-6*abs(reference)','lmm-public-r':'per-case/per-quantity policy stored in fixture',
 'lmm-expanded-r':'likelihood: 5e-6 + 1e-9*abs(reference); estimates versus R bobyqa: 1e-5 + 1e-5*abs(reference); other optimizers: 1e-3 + 3e-4*abs(reference)',
 'logistic-mixed-r':'inference: 2e-5 + 2e-3*abs(reference); likelihood/AIC/BIC: 1e-6 + 1e-8*abs(reference); others: 2e-5 + 2e-4*abs(reference)',
 'stress-r':'per-case absolute+relative; absolute scales with response_scale for estimates and its square for covariance; dimensionless quantities unscaled'}
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def compact_ref(ref):return {k:v for k,v in ref.items() if k not in ['result','results','expected']}
report={'schema_version':1,'purpose':'Reference provenance index; test sources and fixtures are authoritative, not a new snapshot of estimates.','fixtures':[]}
for stem,generator in GENERATORS.items():
 path=ROOT/'tests/fixtures'/f'{stem}.json';data=json.loads(path.read_text());cases=[]
 for c in data['cases']:
  inputs={k:c[k] for k in ['model','x','y','groups','group_labels','terms','weights','offset','reml','quadrature_points'] if k in c}
  expected=c.get('expected',c.get('reference',{}).get('result',{}))
  references=c.get('references',{})
  if references:expected=next(iter(references.values()))['result']
  metadata={k:c[k] for k in ['name','dataset','source','tolerance','response_scale'] if k in c}
  if isinstance(c.get('reference'),dict):metadata['reference']=compact_ref(c['reference'])
  if references:metadata['references']={k:compact_ref(v) for k,v in references.items()}
  if 'nlme_reference' in c:metadata['nlme_reference']=compact_ref(c['nlme_reference'])
  cases.append({**metadata,'inputs_sha256':hashlib.sha256(json.dumps(inputs,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
    'n':len(c['y']),'p':len(c['x'][0]),'compared_quantities':list(expected)})
 report['fixtures'].append({'path':str(path.relative_to(ROOT)),'sha256':sha(path),'generator':generator,'generator_sha256':sha(ROOT/generator),
  'software_recorded_in_fixture':{k:v for k,v in data.items() if k!='cases'},'tolerance_policy':POLICY[stem],
  'model_specification':'Explicit saved matrices/vectors/options in fixture; generator records reference formula, family, seed and controls.',
  'cases':cases})
output=ROOT/'docs/verifications/reference-manifest.json';text=json.dumps(report,indent=2)+'\n'
if '--check' in sys.argv:
 if output.read_text()!=text:raise SystemExit('Reference manifest drifted; regenerate scripts/reference-manifest.py and review evidence changes')
 print('Reference provenance hashes verified')
else:output.write_text(text);print('Reference provenance manifest generated')
