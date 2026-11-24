from common import *
from sklearn.svm import SVC
CS=[.1,1,10,100]
GS=['scale',.01,.1,1]
def run(scale=True,x=X):
 rows=[]
 for kernel in ['linear','rbf']:
  for c in CS:
   for g in (GS if kernel=='rbf' else ['scale']):
    m,_,_=evaluate(pipe(SVC(kernel=kernel,C=c,gamma=g),scale),True,x)
    rows.append(dict(kernel=kernel,C=c,gamma=g,**m))
 return pd.DataFrame(rows)
def model(row,scale=True): return pipe(SVC(kernel=row['kernel'],C=float(row['C']),gamma=row['gamma'] if row['gamma'] in ['scale','auto'] else float(row['gamma'])),scale)

