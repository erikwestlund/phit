"""Render saved benchmark measurements; this never runs benchmark workers."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LANGUAGES = ('R', 'statsmodels', 'PHP')
COLORS = ('#0072B2', '#D55E00', '#009E73')
TITLES = {'ols': 'OLS', 'logistic': 'Logistic',
          'probit': 'Probit', 'poisson': 'Poisson', 'lmm': 'Mixed model'}
HERE = Path(__file__).resolve().parent


def render_charts(report, model):
    cases = sorted(report['cases'], key=lambda case: (case['n'], case['p']))
    output = HERE / 'figures'
    output.mkdir(exist_ok=True)
    style = {'font.family': 'DejaVu Sans', 'font.size': 11,
             'axes.spines.top': False, 'axes.spines.right': False,
             'axes.spines.left': False, 'axes.edgecolor': '#CBD5E1',
             'text.color': '#253247', 'axes.labelcolor': '#253247',
             'xtick.color': '#253247', 'ytick.color': '#536174',
             'svg.fonttype': 'none', 'svg.hashsalt': 'php-stats-benchmarks'}
    with plt.rc_context(style):
        for metric in ('fit-time', 'memory'):
            figure, axes = plt.subplots((len(cases)+1)//2, 2, figsize=(10, 7.5), squeeze=False)
            for axis, case in zip(axes.flat, cases):
                implementations = [case['implementations'][name] for name in LANGUAGES]
                values = [(fit['median_fit_seconds']*1000 if metric == 'fit-time'
                           else fit['peak_rss_bytes']/2**20 if fit['peak_rss_bytes'] else None)
                          for fit in implementations]
                bars = axis.bar(range(3), [v or 0 for v in values], width=0.58,
                                color=COLORS, alpha=0.85, zorder=2)
                maximum = max((v or 0 for v in values), default=0)
                if metric == 'fit-time':
                    for j, fit in enumerate(implementations):
                        samples = np.asarray(fit['fit_seconds'])*1000
                        maximum = max(maximum, float(samples.max()))
                        # Small horizontal offsets expose repeated identical timer readings.
                        axis.scatter(j+np.linspace(-0.16,0.16,len(samples)), samples,
                                     s=15, color='#17202D', alpha=0.8, zorder=3)
                labels = ['not recorded' if v is None else
                          '0*' if v == 0 and metric == 'fit-time' else
                          (f'{v:.3f}' if v < 1 else f'{v:.1f}').rstrip('0').rstrip('.')
                          for v in values]
                axis.bar_label(bars, labels=labels, padding=7, fontsize=11, fontweight='bold')
                axis.set_ylim(0, (maximum or 1)*1.28)
                axis.set_xticks(range(3), LANGUAGES)
                axis.tick_params(axis='both', length=0, pad=7)
                axis.set_ylabel('Full fit (ms)' if metric == 'fit-time' else 'Peak RSS (MiB)')
                axis.set_title(f"{case['n']:,} observations · {case['p']} coefficients", loc='left', pad=15, fontsize=12)
                axis.set_axisbelow(True)
                axis.grid(axis='y', color='#E2E8F0', linewidth=0.8)
            for axis in list(axes.flat)[len(cases):]:
                axis.set_visible(False)
            title = 'Full-fit time' if metric == 'fit-time' else 'Peak process memory'
            figure.suptitle(f'{TITLES[model]} · {title}', x=0.08, ha='left', fontsize=19, fontweight='bold')
            subtitle = (f"Bars: median of {report['repeats']} fits · Dots: individual fits · Lower is faster"
                        if metric == 'fit-time' else 'Entire worker process, including inputs, runtime and outputs · Lower uses less memory')
            figure.text(0.08, 0.92, subtitle, fontsize=10)
            footer = 'Each panel uses its own linear scale. Coefficient counts include the intercept.'
            if metric == 'fit-time':
                footer += '\nStartup and CSV loading excluded. Zero (*) can reflect timer resolution.'
            else:
                footer += '\nProcess memory is not the allocation count of the Rust solver.'
            figure.text(0.08, 0.025, footer, fontsize=9, color='#536174')
            figure.subplots_adjust(top=0.84, bottom=0.13, left=0.09, right=0.97, hspace=0.57, wspace=0.28)
            stem = output/f'{model}-{metric}'
            svg = stem.with_suffix('.svg')
            figure.savefig(svg, metadata={'Date': None})
            svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
            figure.savefig(stem.with_suffix('.png'), dpi=160)
            plt.close(figure)


if __name__ == '__main__':
    for model in TITLES:
        render_charts(json.loads((HERE/f'{model}-results.json').read_text()), model)
        print(f'{model}: fit-time and memory charts refreshed')
