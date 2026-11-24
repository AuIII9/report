from common import *
import classify,regress
from sklearn.linear_model import Lasso

def supplement():
 c=pd.read_csv(ROOT/'results/classification.csv'); r=pd.read_csv(ROOT/'results/regression.csv'); a=pd.read_csv(ROOT/'results/ablations.csv'); u=pd.read_csv(ROOT/'results/unscaled.csv')
 text='\n## 补充：按实际结果作出判断\n'
 for k in ['linear','rbf']:
  b=c[c.kernel==k].sort_values('auc').iloc[-1]; n=u[u.kernel==k].sort_values('auc').iloc[-1]
  text+=f'\n{k}：标准化最优 C={b.C}、AUC={b.auc:.4f}；不标准化最优 C={n.C}、AUC={n.auc:.4f}。最优 C 的变化说明其含义依赖特征尺度。\n'
 for name in ['SVM','Tree','Ridge']:
  t=a[(a.experiment=='features')&(a.model==name)]; metric='auc' if name=='SVM' else 'r2'; vals=t.set_index('setting')[metric]
  text+=f'\n{name} 删除 ca/thal 前 {metric}={vals["all"]:.4f}，删除后 {vals["drop ca thal"]:.4f}，变化 {vals["drop ca thal"]-vals["all"]:+.4f}。\n'
 b=r[r.model=='tree'].sort_values('r2').iloc[-1]; p=r[r.model=='pruned'].sort_values('r2').iloc[-1]
 text+=f'\n本次预剪枝最佳 R²={b.r2:.4f}，后剪枝最佳 R²={p.r2:.4f}；当前搜索范围内预剪枝更好，不能据此证实后剪枝总能避免剪早而提高成绩。\n'
 for i in range(len(r)):
  found=False
  for j in range(i+1,len(r)):
   q,s=r.iloc[i],r.iloc[j]
   if q.r2>s.r2 and q.mae>s.mae:
    text+=f'\n指标逆序的具体例子：{q.model}({q.parameter}) R²={q.r2:.4f}、MAE={q.mae:.4f}；{s.model}({s.parameter}) R²={s.r2:.4f}、MAE={s.mae:.4f}。在这两者中按 R² 选前者，按 MAE 选后者。全体候选按 R² 的最佳为 {r.loc[r.r2.idxmax(),"model"]}({r.loc[r.r2.idxmax(),"parameter"]})，按 MAE 的最佳为 {r.loc[r.mae.idxmin(),"model"]}({r.loc[r.mae.idxmin(),"parameter"]})；局部逆序不一定改变全局最佳。\n'; found=True; break
  if found: break
 corr=X.corr(); pairs=[(abs(corr.iloc[i,j]),X.columns[i],X.columns[j],corr.iloc[i,j]) for i in range(13) for j in range(i)]
 text+='\n相关性最大的五对特征：'+str(sorted(pairs,reverse=True)[:5])+'。相关性为探索性全表描述，不参与预处理拟合。\n'
 rows=[]
 for alpha in [.01,.1,1,10]:
  q=pipe(Lasso(alpha=alpha,max_iter=10000)).fit(X,y); m,_,_=evaluate(pipe(Lasso(alpha=alpha,max_iter=10000)))
  rows.append(dict(alpha=alpha,zero_coefficients=int(np.sum(q[-1].coef_==0)),**m))
 text+='\n为核验讲义关于 Lasso 的论断，增加辅助实验（不替代 Ridge 主线）：\n\n'+pd.DataFrame(rows).to_markdown(index=False)+'\n'
 pr=regress.model(p).fit(X,y)[-1]; nodes=[0,pr.tree_.children_left[0],pr.tree_.children_right[0]]
 text+='\n剪枝树前两层的实际规则：\n'
 for node in nodes:
  if node>=0 and pr.tree_.feature[node]>=0:
   f=X.columns[pr.tree_.feature[node]]; text+=f'\n- 节点 {node}: {f} <= {pr.tree_.threshold[node]:.3f}，样本数 {pr.tree_.n_node_samples[node]}。这是按该检查指标将病例分组，叶值表示组内平均严重程度，不能解释为因果或独立诊断标准。\n'
 (ROOT/'report.md').write_text((ROOT/'report.md').read_text(encoding='utf-8')+text,encoding='utf-8')


def wide_tables():
 from sklearn.dummy import DummyClassifier,DummyRegressor
 c=pd.read_csv(ROOT/'results/classification.csv'); r=pd.read_csv(ROOT/'results/regression.csv',keep_default_na=False)
 text='\n## 分类与回归汇总（宽表）\n'
 for cls in [True,False]:
  rows=[]
  if cls:
   specs=[(k,classify.model(dict(kernel=k,C=1,gamma='scale')),classify.model(c[c.kernel==k].sort_values('auc').iloc[-1])) for k in ['linear','rbf']]
  else:
   specs=[('Tree',regress.tree(3),regress.model(r[r.model.isin(['tree','pruned'])].sort_values('r2').iloc[-1])),('Ridge',pipe(regress.Ridge(alpha=1)),regress.model(r[r.model=='ridge'].sort_values('r2').iloc[-1]))]
  dummy=pipe(DummyClassifier(strategy='most_frequent') if cls else DummyRegressor()); specs.append(('Naive reference',dummy,dummy))
  for name,base,best in specs:
   bm,_,_=evaluate(base,cls); om,_,_=evaluate(best,cls)
   metrics=['accuracy','precision','recall','f1','auc'] if cls else ['r2','mae','rmse']
   rows.append(dict(model=name,**{'baseline_'+k:bm[k] for k in metrics},**{'optimal_'+k:om[k] for k in metrics}))
  text+='\n'+pd.DataFrame(rows).to_markdown(index=False,floatfmt='.4f')+'\n\n我从这张表里读出：调参提升幅度有限，但模型明显优于朴素参考，比较时必须固定指标口径。\n'
 with (ROOT/'report.md').open('a',encoding='utf-8') as f: f.write(text)
