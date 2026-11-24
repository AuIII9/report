from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *
ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/'assignment4-data'
df=pd.read_csv(DATA/'heart_cleveland.csv'); X=df.drop(columns='num'); y=df.num.to_numpy(); z=(y>0).astype(int)
folds=json.loads((DATA/'splits.json').read_text())['cleveland']['folds']
assert sorted(sum(folds,[]))==list(range(len(y)))
cv=[(np.setdiff1d(np.arange(len(y)),t),np.array(t)) for t in folds]
def pipe(m,scale=True):
 return make_pipeline(SimpleImputer(strategy='median'),*([StandardScaler()] if scale else []),m)
def evaluate(p,cls=False,x=X,splits=cv):
 target=z if cls else y; rows=[]; pred=np.zeros(len(y)); scores=np.zeros(len(y))
 for tr,te in splits:
  q=clone(p).fit(x.iloc[tr],target[tr]); a=q.predict(x.iloc[te]); pred[te]=a
  if cls:
   s=q.decision_function(x.iloc[te]) if hasattr(q,'decision_function') else q.predict_proba(x.iloc[te])[:,1]; scores[te]=s
   st=q.decision_function(x.iloc[tr]) if hasattr(q,'decision_function') else q.predict_proba(x.iloc[tr])[:,1]
   r=dict(accuracy=accuracy_score(z[te],a),precision=precision_score(z[te],a,zero_division=0),recall=recall_score(z[te],a),f1=f1_score(z[te],a),auc=roc_auc_score(z[te],s),train_auc=roc_auc_score(z[tr],st))
  else:
   r=dict(r2=r2_score(y[te],a),mae=mean_absolute_error(y[te],a),mse=mean_squared_error(y[te],a))
   m=q.steps[-1][1]
   if hasattr(m,'get_depth'): r.update(depth=m.get_depth(),leaves=m.get_n_leaves())
  rows.append(r)
 result=pd.DataFrame(rows).mean().to_dict()
 if not cls: result['rmse']=np.sqrt(result.pop('mse'))
 return result,pred,scores
