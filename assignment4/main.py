"""Run: python assignment4/main.py; dependencies pinned in requirements.txt."""
from common import *
import classify,regress,ablations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.dummy import DummyClassifier,DummyRegressor
import subprocess,sys,importlib.metadata

def table(d): return d.to_markdown(index=False,floatfmt='.4f')
def main():
 (ROOT/'results').mkdir(exist_ok=True); (ROOT/'figures').mkdir(exist_ok=True)
 c=classify.run(); r=regress.run(); c.to_csv(ROOT/'results/classification.csv',index=False); r.to_csv(ROOT/'results/regression.csv',index=False)
 cb=c.loc[c.auc.idxmax()]; rb=r[r.model=='ridge'].sort_values('r2').iloc[-1]
 a=ablations.run(cb,rb); a.to_csv(ROOT/'results/ablations.csv',index=False)
 refout=subprocess.check_output([sys.executable,str(DATA/'baselines.py'),'--json'],cwd=DATA).decode('utf-8').replace('\r\n','\n')
 ref=json.loads(refout[refout.index('{\n'):]); checks=[]; summaries=[]
 specs=[('SVM (linear)',classify.model(dict(kernel='linear',C=1,gamma='scale')),True,c[c.kernel=='linear'].sort_values('auc').iloc[-1]),('SVM (RBF)',classify.model(dict(kernel='rbf',C=1,gamma='scale')),True,c[c.kernel=='rbf'].sort_values('auc').iloc[-1]),('Ridge (alpha=1)',pipe(regress.Ridge(alpha=1)),False,rb),('Regression Tree (d=3)',regress.tree(3),False,r[r.model.isin(['tree','pruned'])].sort_values('r2').iloc[-1])]
 for name,p,cls,best in specs:
  base,_,_=evaluate(p,cls); opt,_,_=evaluate(classify.model(best) if cls else regress.model(best),cls)
  for metric,value in ref['cleveland/'+('cls/' if cls else 'reg/')+name].items():
   checks.append(dict(model=name,metric=metric,baseline=value,mine=base[metric],difference=base[metric]-value))
  for metric in (['accuracy','precision','recall','f1','auc'] if cls else ['r2','mae','rmse']): summaries.append(dict(model=name,metric=metric,baseline=base[metric],optimal=opt[metric]))
 checks=pd.DataFrame(checks); assert checks.difference.abs().max()<.01
 checks.to_csv(ROOT/'results/baseline_check.csv',index=False)
 cm,cp,cs=evaluate(classify.model(cb),True); rm,rp,_=evaluate(regress.model(rb))
 detail=pd.DataFrame(classification_report(z,cp,output_dict=True,zero_division=0)).T
 thresholds=pd.DataFrame([dict(threshold=t,accuracy=accuracy_score(z,rp>t),f1=f1_score(z,rp>t),auc=roc_auc_score(z,rp)) for t in [.3,.5,.7,1.]])
 thresholds.to_csv(ROOT/'results/thresholds.csv',index=False)
 coef=[]
 for alpha in [.000001,.01,.1,1,10,100,1000]:
  q=pipe(regress.Ridge(alpha=alpha)).fit(X,y)
  coef.append(dict(alpha=alpha,**dict(zip(X.columns,q[-1].coef_))))
 co=pd.DataFrame(coef); co.to_csv(ROOT/'results/coefficients.csv',index=False)
 def save(name): plt.tight_layout(); plt.savefig(ROOT/'figures'/name,dpi=160); plt.close()
 df.num.value_counts().sort_index().plot.bar(); plt.xlabel('num'); plt.ylabel('Count'); save('labels.png')
 plt.scatter(df.oldpeak,y,alpha=.4); plt.xlabel('oldpeak'); plt.ylabel('num'); save('feature.png')
 grid=c[c.kernel=='rbf'].pivot(index='C',columns='gamma',values='auc'); plt.imshow(grid,aspect='auto'); plt.xticks(range(len(grid.columns)),grid.columns); plt.yticks(range(len(grid)),grid.index); plt.xlabel('gamma'); plt.ylabel('C'); plt.colorbar(label='CV AUC'); save('rbf.png')
 plt.plot(range(len(r[r.model=='tree'])),r[r.model=='tree'].r2,'o-'); plt.xticks(range(len(regress.DEPTHS)),regress.DEPTHS); plt.xlabel('max_depth'); plt.ylabel('CV R2'); save('depth.png')
 pr=r[r.model=='pruned'].sort_values('r2').iloc[-1]; pt=regress.model(pr).fit(X,y)
 plt.figure(figsize=(22,10)); plot_tree(pt[-1],feature_names=list(X.columns),filled=True,rounded=True,fontsize=6); save('tree.png')
 plt.imshow(confusion_matrix(z,cp)); plt.colorbar(); plt.xlabel('Predicted'); plt.ylabel('True'); save('confusion.png')
 dummyc,_,_=evaluate(pipe(DummyClassifier(strategy='most_frequent')),True); dummyr,_,_=evaluate(pipe(DummyRegressor()),False)
 splits=a[a.experiment=='split'].r2; un=pd.read_csv(ROOT/'results/unscaled.csv')
 lin=c[c.kernel=='linear'].sort_values('auc').iloc[-1]; rbf=c[c.kernel=='rbf'].sort_values('auc').iloc[-1]
 analysis=f'''# 讨论题

## 1. 线性核与 RBF
线性核最优 AUC 为 {lin.auc:.4f}，RBF 为 {rbf.auc:.4f}。本次较优核是 {cb.kernel}，其 C={cb.C}、gamma={cb.gamma}。非线性并不保证收益，局部边界可能放大小样本噪声。RBF 的核值 exp(-gamma*距离平方) 在 gamma 增大时迅速衰减，训练样本间的联系变弱。扫描表同时报告训练 AUC 与验证 AUC，可直接检查拟合和泛化的差距。核更换与调参的收益应分别用总表中的最优差值和基线到最优差值衡量，不能将两者混为一谈。

## 2. 两条任务线
岭回归最优 R² 为 {rb.r2:.4f}，树族最优 R² 为 {r[r.model!='ridge'].r2.max():.4f}。分类最优 AUC 为 {cb.auc:.4f}。这些数值不能跨任务直接比较大小。若线性方法在两条任务线上均有竞争力，说明当前特征中的主要信号可由较简单的关系表达。树的局部分割会减少叶中样本数，均值估计因此更不稳定。该解释只适用于当前样本和特征，不能推出真实疾病机制必然线性。

## 3. 回归作为分类器
岭回归折外预测作为连续分数的 AUC 为 {roc_auc_score(z,rp):.4f}，SVM 折外 AUC 为 {roc_auc_score(z,cs):.4f}。表中的多个阈值显示，同一排序可以产生不同的 accuracy 与 F1。SVM 折外 accuracy={accuracy_score(z,cp):.4f}、F1={f1_score(z,cp):.4f}。因此“排序不逊于分类器”与“某个阈值的类别判断更差”可以同时成立。回归目标 num 与 num>0 相关，所以无需直接拟合二分类标签也能得到有用排序。其局限是平方损失更关注等级误差，未直接优化零与非零的边界，且这些阈值是探索结果，部署前应在独立数据上选择。

{table(thresholds)}
'''
 (ROOT/'analysis.md').write_text(analysis,encoding='utf-8')
 report=f'''# Assignment 4 实验报告

## 设计与复现
运行 `python assignment4/main.py`，或在本目录运行 `python main.py`。课程数据放在同级的 assignment4-data。使用固定五折，种子 42；单次划分实验种子 0–9。中位数填补与标准化均在 Pipeline 中，按训练折拟合。离散编码沿用基线数值表示，不额外做独热编码，以便对齐。分类主指标 AUC，回归主指标 R²。扫描与择优使用相同五折，因此最优结果存在选择乐观偏差，不是独立测试性能；本报告按课程协议进行探索，不作临床部署结论。剪枝候选从训练折路径汇总，候选搜索也属于探索过程。

分类是 num>0，回归是 num；num 不是 ca。若改为五分类，会忽略有序距离且稀有等级样本更少。

## 数据探索
303 行，缺失 {int(X.isna().sum().sum())} 个单元格。用训练折中位数保留样本，避免删除带来的选择偏差。

{table(X.describe().T.reset_index())}

![标签分布](figures/labels.png)
![特征与标签](figures/feature.png)

## 基线复现
基线由课程脚本独立运行，自己的代码逐折拟合并计算；没有复制脚本作为实验实现。RMSE 是平均折 MSE 的平方根，与课程脚本一致。

{table(checks)}

全部差值小于 0.01，支持当前评估流程正确。

## 总表：固定基线与扫描最优
{table(pd.DataFrame(summaries))}

朴素分类参考：{dummyc}；朴素回归参考：{dummyr}。训练均值在测试折的 R² 不保证恰好为零，不能把全表均值的理想参考误写成实际测试结果。

我从表里读出：调参收益应逐模型逐指标比较，复杂模型并不必然领先。

## 分类扫描
{table(c)}

![RBF 热力图](figures/rbf.png)

C 增大意味着错分惩罚增强、正则化减弱；并不是更不能容忍越界的参数值是小 C。对固定核比较训练与验证差距，可区分容量不足和过拟合。最佳设置为 {cb.to_dict()}。

逐类指标采用汇总折外预测，accuracy 是整体指标，AUC 是二分类排序指标，不伪造逐类 accuracy。敏感性等于正类 recall，特异性等于负类 recall；precision 的分母是预测为该类的样本。

{table(detail.reset_index())}

混淆矩阵（行真值，列预测）：{confusion_matrix(z,cp).tolist()}。
![混淆矩阵](figures/confusion.png)

## 回归扫描及剪枝
{table(r)}

![树深扫描](figures/depth.png)
![剪枝后的树](figures/tree.png)

预剪枝限制生长，可能错过需要多次分裂才能显现的收益；后剪枝先长树再按复杂度代价裁剪，但不保证更好。叶节点增多使每叶样本减少，局部均值方差增大，深树在未见样本上容易产生大误差。树图中的前两层以特征名称和阈值给出了分组规则；涉及 ca/thal 时依赖昂贵检查，涉及 oldpeak 时反映运动 ST 段变化，不能解释为因果效应。

以下分别按 R² 与 MAE 排序，尤其比较深度 3 与 5：平方误差对少数大误差更敏感，MAE 改善不保证 R² 改善。

{table(r.sort_values('r2',ascending=False))}

{table(r.sort_values('mae'))}

## 正则化系数
{table(co)}

接近零的 alpha 近似 OLS；增大 alpha 收缩系数，但 Ridge 不执行 Lasso 式稀疏选择。系数来自全表拟合，仅用于解释，不用于计算测试指标。相关特征会分摊信号，系数大小不能当作独立因果效应；Lasso 在相关组间的选择可能不稳定。本实验主线选择 Ridge。

## 四项消融
{table(a)}

去除 ca/thal 的变化应按模型分别比较；下降表示这些检查提供额外信息，却不代表初筛能够合法获得这些信息。树按排序阈值分裂，因此正比例缩放通常不改变分组；SVM 的距离和惩罚尺度会改变。

不标准化的完整 C 扫描：
{table(un)}

10 次单划分 R² 最小 {splits.min():.4f}、最大 {splits.max():.4f}、标准差（ddof=1）{splits.std():.4f}；固定五折树深 3 的 R²={r[(r.model=='tree')&(r.parameter=='3')].r2.iloc[0]:.4f}。仅凭一次划分可能选中偶然有利的测试集。多项式容量实验增加交互与平方项，也增加估计方差；强正则化缓解但不能保证收益，与深树的方差问题相通。

## 讨论题
{analysis}

## 实验日志与可检验假设
1. 非线性未必改善：线性最优 AUC={lin.auc:.4f}，RBF={rbf.auc:.4f}。假设小样本下局部边界增加方差；验证方式为扫描 gamma 并比较训练/验证 AUC，见分类扫描表。
2. 深树未必更好：深度 3 R²={r[(r.model=='tree')&(r.parameter=='3')].r2.iloc[0]:.4f}，无限深 R²={r[(r.model=='tree')&(r.parameter=='None')].r2.iloc[0]:.4f}。假设小叶子估计不稳定；通过叶数和深度扫描检查复杂度变化，后续可进一步统计叶内样本量。
3. 同一模型单划分波动：R² 范围 {splits.min():.4f}–{splits.max():.4f}。假设测试集中严重等级的构成影响平方误差；10 个种子证实波动存在，进一步验证原因应按等级分解误差，不能把波动本身当成机制证明。

## AI 使用声明与来源
本项目由 OpenAI Codex 辅助编写实验代码、运行、生成图表与撰写报告；用户应自行审阅理解后提交。未虚构用户独立完成的验证经历。已通过独立运行课程基线、固定折覆盖断言和指标差值断言验证程序；纠正了把训练均值参考 R² 必定写成 0 的误导性简化。来源为课程 Assignment 4 PDF、assignment4-data/README.md 与 baselines.py，以及本次 Codex 对话。代码不使用禁止的集成或深度模型。
'''
 (ROOT/'report.md').write_text(report,encoding='utf-8')
 (ROOT/'requirements.txt').write_text('\n'.join(k+'=='+importlib.metadata.version(k) for k in ['numpy','pandas','scikit-learn','matplotlib','tabulate'])+'\n')
 print('DONE baseline max difference:',checks.difference.abs().max())
if __name__=='__main__':
 main()
 from supplement import supplement
 supplement()


 from supplement import wide_tables; wide_tables()
