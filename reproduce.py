"""Recreate paper figures and complete result tables from saved scalar records."""
import argparse,os,subprocess,sys,tempfile
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--output',type=Path,default=Path('generated'))
 p.add_argument('--tables-only',action='store_true')
 a=p.parse_args();root=Path(__file__).resolve().parent
 out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
 env={**os.environ,'REPRODUCTION_OUTPUT':str(out),'PYTHONDONTWRITEBYTECODE':'1'}
 scripts=['tables/build_tables.py'] if a.tables_only else ['figures/build_figures.py','figures/build_qualitative.py','tables/build_tables.py']
 with tempfile.TemporaryDirectory(prefix='aig-figures-') as cache:
  env['MPLCONFIGDIR']=cache
  for script in scripts:subprocess.run([sys.executable,str(root/script)],env=env,check=True)
 print('Wrote '+str(out))
if __name__=='__main__':main()
