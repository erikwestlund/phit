"""Compatibility bridge for existing Jupyter pages; all drawing is R/ggplot2."""
import json,os,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
def render_charts(report,model):
    saved=json.loads((HERE/f'{model}-results.json').read_text())
    if saved!=report:
        raise ValueError('Plot input differs from saved benchmark evidence')
    subprocess.run([os.environ.get('RSCRIPT','Rscript'),str(HERE.parent/'figures/render.R'),model],check=True)
if __name__=='__main__':
    for model in ['ols','logistic','probit','poisson','lmm']:
        render_charts(json.loads((HERE/f'{model}-results.json').read_text()),model)
