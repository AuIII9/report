from common import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
DEPTHS=[1,2,3,4,5,6,8,10,None]
def tree(depth=None,alpha=0): return pipe(DecisionTreeRegressor(max_depth=depth,ccp_alpha=alpha,random_state=42),False)
def run():
 rows=[]
 for d in DEPTHS:
  m,_,_=evaluate(tree(d)); rows.append(dict(model='tree',parameter=str(d),**m))
 # Candidate paths use training folds only; held-out labels never enter a path.
 paths=[]
 for tr,te in cv:
  a=SimpleImputer(strategy='median').fit_transform(X.iloc[tr])
  paths.extend(DecisionTreeRegressor(random_state=42).cost_complexity_pruning_path(a,y[tr]).ccp_alphas)
 for a in np.unique(np.quantile(paths,np.linspace(0,1,12))):
  m,_,_=evaluate(tree(alpha=a)); rows.append(dict(model='pruned',parameter=a,**m))
 for a in [.000001,.01,.1,1,10,100,1000]:
  m,_,_=evaluate(pipe(Ridge(alpha=a))); rows.append(dict(model='ridge',parameter=a,**m))
 return pd.DataFrame(rows)
def model(r):
 if r['model']=='ridge': return pipe(Ridge(alpha=float(r['parameter'])))
 if r['model']=='pruned': return tree(alpha=float(r['parameter']))
 return tree(None if r['parameter']=='None' else int(r['parameter']))
