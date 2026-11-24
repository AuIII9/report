from common import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
import classify,regress
def run(cbest,rbest):
 rows=[]
 for name,p,cls in [('SVM',classify.model(cbest),True),('Tree',regress.tree(3),False),('Ridge',regress.model(rbest),False)]:
  for remove in [False,True]:
   m,_,_=evaluate(p,cls,X.drop(columns=['ca','thal']) if remove else X)
   rows.append(dict(experiment='features',model=name,setting='drop ca thal' if remove else 'all',**m))
 unscaled=classify.run(False); unscaled.to_csv(ROOT/'results/unscaled.csv',index=False)
 for seed in range(10):
  tr,te=train_test_split(np.arange(len(y)),test_size=.2,random_state=seed)
  m,_,_=evaluate(regress.tree(3),splits=[(tr,te)])
  rows.append(dict(experiment='split',model='Tree',setting=seed,**m))
 for a in [10,100,1000,10000]:
  p=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),PolynomialFeatures(2,include_bias=False),StandardScaler(),Ridge(alpha=a))
  m,_,_=evaluate(p); rows.append(dict(experiment='capacity',model='Polynomial Ridge',setting=a,**m))
 return pd.DataFrame(rows)
