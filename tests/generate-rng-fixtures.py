#!/usr/bin/env python3
"""Independent Python ChaCha20/PCG seed expansion plus base-R inverse CDFs.

No native extension or rand library is used. Mirrors the documented pinned
rand_core 0.9.5 seed_from_u64 and rand 0.9.5 Open01 mapping, and implements the
ChaCha20 ARX block equations independently. Crosses block/refill boundaries.
"""
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MASK32 = (1 << 32)-1
MASK64 = (1 << 64)-1


def rotate(v, k):
    return ((v << k) | (v >> (32-k))) & MASK32


def seed_key(seed):
    key = []
    for _ in range(8):
        seed = (seed*6364136223846793005 + 11634580027462260723) & MASK64
        x = (((seed >> 18) ^ seed) >> 27) & MASK32
        k = seed >> 59
        key.append(((x >> k) | (x << ((32-k) % 32))) & MASK32)
    return key


def block(key, counter):
    initial = [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574,
               *key, counter & MASK32, counter >> 32, 0, 0]
    x = initial.copy()

    def quarter(a, b, c, d):
        x[a] = (x[a]+x[b]) & MASK32; x[d] = rotate(x[d]^x[a], 16)
        x[c] = (x[c]+x[d]) & MASK32; x[b] = rotate(x[b]^x[c], 12)
        x[a] = (x[a]+x[b]) & MASK32; x[d] = rotate(x[d]^x[a], 8)
        x[c] = (x[c]+x[d]) & MASK32; x[b] = rotate(x[b]^x[c], 7)

    for _ in range(10):
        for args in [(0,4,8,12),(1,5,9,13),(2,6,10,14), (3,7,11,15),
                     (0,5,10,15),(1,6,11,12),(2,7,8,13),(3,4,9,14)]:
            quarter(*args)
    return [(a+b) & MASK32 for a,b in zip(x,initial)]


def uniforms(seed, n):
    key = seed_key(seed)
    words = []
    for i in range((n+7)//8):
        words.extend(block(key,i))
    result = []
    for i in range(n):
        word = words[2*i] | (words[2*i+1] << 32)
        # Midpoints of the 52-bit uniform cells: strictly between zero and one.
        result.append(((word >> 12)+.5)*2**-52)
    return result


def main():
    cases = []
    for seed in [0,42,20261012,2**63-1]:
        cases.append(dict(name=f'uniform_seed_{seed}', distribution='uniform', operation='rng',
                          seed=seed, n=96, params={}, input=uniforms(seed,96)))
    families = dict(normal=dict(mean=3,sd=2), t=dict(df=12), chisquare=dict(df=4),
        f=dict(df1=4,df2=40), beta=dict(alpha=2,beta=5), gamma=dict(shape=2,scale=3),
        exponential=dict(rate=.5), lognormal=dict(meanlog=2,sdlog=.5), logistic=dict(location=5,scale=2),
        binomial=dict(size=20,prob=.3), poisson=dict(lambda_=12), nbinom=dict(size=3,prob=.4),
        nbinom_mu=dict(size=3,mu=4.5))
    families['poisson'] = {'lambda':families['poisson']['lambda_']}
    for family, params in families.items():
        cases.append(dict(name=family,distribution=family,operation='rng',seed=42,n=96,
                          params=params,input=uniforms(42,96)))
    for seed in [0,20261012,2**63-1]:
        cases.append(dict(name=f'normal_seed_{seed}',distribution='normal',operation='rng',seed=seed,
                          n=96,params=families['normal'],input=uniforms(seed,96)))
    for name, family, params in [('zero_poisson','poisson',{'lambda':0}),
        ('zero_binomial','binomial',{'size':0,'prob':.4}),
        ('certain_binomial','binomial',{'size':20,'prob':1}),
        ('zero_nb','nbinom',{'size':3,'prob':1}), ('zero_nb_mu','nbinom_mu',{'size':3,'mu':0})]:
        cases.append(dict(name=name,distribution=family,operation='rng',seed=42,n=16,
                          params=params,input=uniforms(42,16)))
    data = dict(generator='tests/generate-rng-fixtures.py + tests/generate-rng-fixtures.R',
        python=platform.python_version(), engine='ChaCha20; rand_core 0.9.5 PCG32 seed_from_u64 expansion; rand 0.9.5 Open01 f64 mapping',
        provenance='Uniforms generated independently in Python from the published block/seed/mapping equations; R quantiles transform those saved uniforms, without calling the extension.',
        tolerance='Uniforms exact bit-for-bit; R quantiles 1e-15 + 1e-12*abs(reference); discrete and point masses exact',cases=cases)
    with tempfile.TemporaryDirectory(prefix='rng-reference-') as directory:
        path = Path(directory)/'uniforms.json'; path.write_text(json.dumps(data,allow_nan=False))
        output = Path(directory)/'quantiles.json'
        subprocess.run([os.environ.get('RSCRIPT','Rscript'),str(ROOT/'tests/generate-rng-fixtures.R'),str(path),str(output)],check=True)
        reference = json.loads(output.read_text())
        data.update(reference=reference['reference'],jsonlite=reference['jsonlite'])
        for case, expected in zip(cases,reference['expected']):
            # Preserve the Python-derived uniform bits and full integer seed;
            # R's JSON writer uses fewer decimal digits and numeric seeds lose
            # precision above 2^53. R only supplies the independent quantiles.
            case['expected'] = case['input'].copy() if case['distribution']=='uniform' else expected
        (ROOT/'tests/fixtures/rng-python-r.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    print(f'{len(cases)} independent seed/quantile cases written')


if __name__=='__main__':
    main()
