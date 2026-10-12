"""Snapshot provenance and lme4 ML expectations for published clustered data."""
import argparse,csv,hashlib,json,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'docs/data/lmm'
SOURCES=[
 dict(name='sleepstudy',package='lme4',group='Subject',outcome='Reaction',
      documentation='https://lme4.github.io/lme4/reference/sleepstudy.html',
      citation='Belenky et al. (2003), Journal of Sleep Research 12, 1–12; doi:10.1046/j.1365-2869.2003.00337.x',
      formula='Reaction ~ Days + (1 | Subject)',terms=['intercept','Days']),
 dict(name='Orthodont',package='nlme',group='Subject',outcome='distance',
      documentation='https://stat.ethz.ch/R-manual/R-devel/library/nlme/html/Orthodont.html',
      citation='Potthoff and Roy (1964), Biometrika 51, 313–326; Pinheiro and Bates (2000), Appendix A.17',
      formula='distance ~ age * female + (1 | Subject)',terms=['intercept','age','female','age × female']),
 dict(name='Machines',package='nlme',group='Worker',outcome='score',
      documentation='https://stat.ethz.ch/R-manual/R-devel/library/nlme/html/Machines.html',
      citation='Milliken and Johnson (1992), Analysis of Messy Data, p.285; Pinheiro and Bates (2000), Appendix A.14',
      formula='score ~ machine_B + machine_C + (1 | Worker)',terms=['intercept','machine B','machine C']),
 dict(name='Hsb82',package='mlmRev',group='school',outcome='mAch',
      documentation='https://cran.r-project.org/web/packages/mlmRev/mlmRev.pdf#page=11',
      citation='Raudenbush and Bryk (2002), Hierarchical Linear Models, chapter 4; High School and Beyond (1982)',
      formula='mAch ~ ses + meanses + catholic + (1 | school)',terms=['intercept','SES','school mean SES','Catholic school']),
]

def make_cases():
 cases=[]
 metadata=json.loads((HERE/'sources.json').read_text())
 for source in metadata['datasets']:
  path=HERE/source['file'];data=list(csv.DictReader(path.open(newline='')))
  assert hashlib.sha256(path.read_bytes()).hexdigest()==source['sha256']
  x=[];y=[];groups=[];labels=[]
  for row in data:
   if source['name']=='sleepstudy':columns=[1.,float(row['Days'])]
   elif source['name']=='Orthodont':
    age=float(row['age']);female=float(row['Sex']=='Female');columns=[1.,age,female,age*female]
   elif source['name']=='Machines':columns=[1.,float(row['Machine']=='B'),float(row['Machine']=='C')]
   else:columns=[1.,float(row['ses']),float(row['meanses']),float(row['sector']=='Catholic')]
   label=row[source['group']]
   if label not in labels:labels.append(label)
   groups.append(labels.index(label));x.append(columns);y.append(float(row[source['outcome']]))
  assert len(y)==source['n'] and len(labels)==source['n_groups']
  cases.append(dict(name=source['name'],x=x,y=y,groups=groups,group_labels=labels,source=source,
                    tolerance=dict(absolute=1e-7,relative=1e-6,
                                   quantity_absolute={k:1e-6 for k in ['fitted','residuals','fixed_fitted','marginal_residuals']},
                                   quantity_relative={'log_likelihood':1e-10,'aic':1e-10})))
 return cases

def reference(case,backend='lme4'):
 with tempfile.TemporaryDirectory()as directory:
  path=Path(directory)/'data.csv'
  with path.open('w',newline='')as stream:
   writer=csv.writer(stream);writer.writerow(['y','group',*[f'x{j}'for j in range(len(case['x'][0]))]])
   for x,y,g in zip(case['x'],case['y'],case['groups']):writer.writerow([y,g,*x])
  result=subprocess.run(['Rscript',str(ROOT/'docs/verifications/lmm.R'),str(path),backend],capture_output=True,text=True,check=True)
  return json.loads(result.stdout)

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--export',action='store_true',help='Refresh CSV snapshots from installed lme4/nlme/mlmRev packages.')
 args=parser.parse_args()
 if args.export:
  versions=json.loads(subprocess.check_output(['Rscript',str(ROOT/'tests/export-lmm-public-data.R')],cwd=ROOT,text=True))
  datasets=[]
  for source in SOURCES:
   path=HERE/f"{source['name']}.csv";data=list(csv.DictReader(path.open(newline='')))
   sizes={}
   for row in data:sizes[row[source['group']]]=sizes.get(row[source['group']],0)+1
   datasets.append(dict(**source,file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        n=len(data),n_groups=len(sizes),minimum_group_size=min(sizes.values()),maximum_group_size=max(sizes.values()),
                        package_version=versions[source['package']]['version'],package_license=versions[source['package']]['license'],
                        package_source=f"https://cran.r-project.org/package={source['package']}",columns=list(data[0]),
                        selection='All original rows; Hsb82 retains only the five columns used by the model.'))
  (HERE/'sources.json').write_text(json.dumps(dict(datasets=datasets,encoding='UTF-8 CSV; row order preserved; explicit indicator columns constructed in the fixture generator'),indent=2)+'\n')
 cases=make_cases()
 for case in cases:
  case['reference']=reference(case)
  case['nlme_reference']=reference(case,'nlme')
  print(case['name'],len(case['y']),'observations;',len(case['group_labels']),'groups; R references saved')
 (ROOT/'tests/fixtures/lmm-public-r.json').write_text(json.dumps(dict(cases=cases),separators=(',',':'),allow_nan=False)+'\n')
