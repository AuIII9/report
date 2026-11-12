"""Run from repository root: python assignment3/main.py."""
from pathlib import Path
import sys
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from glucose_env import GlucoseEnv
from model_based import ModelBasedAgent
from q_learning import QLearningAgent
S = ['B'+str(i) for i in range(1,10)]
A = ['w','g','i1','i2']
ROOT = Path(__file__).resolve().parent
FIG = ROOT/'figures'
FIG.mkdir(exist_ok=True)

def collect():
    rng = random.Random(2026)
    data = []
    for i in range(2000):
        env = GlucoseEnv(seed=2026+i)
        s, ep = env.reset(), []
        while True:
            a = rng.choice(env.actions)
            ns,r,d,_ = env.step(a)
            ep.append((s,a,r,ns,d)); s=ns
            if d: break
        data.append(ep)
    return data

def evaluate(agent):
    returns, target, steps, failures = [], 0, 0, 0
    for i in range(500):
        env = GlucoseEnv(seed=100000+i)
        s, total = env.reset(), 0
        while True:
            s,r,d,_ = env.step(agent.get_policy(s))
            total += r; target += s=='B3'; steps += 1
            if d:
                failures += s=='B1'
                break
        returns.append(total)
    return [np.mean(returns), target/steps, failures/500]

def table(headers, rows):
    return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+ '\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)+'\n'

def main():
    data = collect()
    mb = ModelBasedAgent(S,A); mb.learn_model(data); mb.run_value_iteration()
    q = QLearningAgent(S,A); q.learn(data)
    metrics = []
    for n in [500,1000,2000]:
        ag = ModelBasedAgent(S,A); ag.learn_model(data[:n]); ag.run_value_iteration()
        metrics.append([str(n)]+evaluate(ag))
    qm = evaluate(q)
    counts = [mb.counts[s,a] for s in S for a in A]
    plt.figure(figsize=(13,4)); plt.bar(range(36), counts)
    plt.xticks(range(36), [s+'/'+a for s in S for a in A], rotation=90)
    plt.ylabel('Visit count'); plt.tight_layout(); plt.savefig(FIG/'visits.png'); plt.close()
    fig,axes = plt.subplots(1,3,figsize=(12,3.5))
    for j,title in enumerate(['Mean return','B3 time fraction','B1 termination rate']):
        axes[j].plot([500,1000,2000],[r[j+1] for r in metrics],marker='o')
        axes[j].axhline(qm[j],color='orange',label='Q (2000)'); axes[j].set_title(title); axes[j].set_xlabel('Episodes'); axes[j].legend()
    fig.tight_layout(); fig.savefig(FIG/'ablation.png'); plt.close(fig)
    curves = {}
    for mode in ['fixed','decay']:
        runs=[]
        for seed in range(5):
            ag=QLearningAgent(S,A); rng=random.Random(50000+seed); returns=[]
            for ep in range(2000):
                env=GlucoseEnv(seed=200000+seed*10000+ep); s=env.reset(); total=0
                while True:
                    a=rng.choice(A) if rng.random()<ag.epsilon else ag.get_policy(s)
                    ns,r,d,_=env.step(a); ag.update(s,a,r,ns,d); total+=r; s=ns
                    if d: break
                returns.append(total)
                if mode=='decay': ag.epsilon=max(ag.epsilon_min,ag.epsilon*ag.epsilon_decay)
            runs.append(np.convolve(returns,np.ones(50)/50,mode='valid'))
        curves[mode]=np.asarray(runs)
    plt.figure(figsize=(9,4))
    for mode,arr in curves.items():
        x=np.arange(50,2001); mean=arr.mean(axis=0); std=arr.std(axis=0)
        plt.plot(x,mean,label=mode); plt.fill_between(x,mean-std,mean+std,alpha=.2)
    plt.xlabel('Episode'); plt.ylabel('Return (50-episode moving mean)'); plt.legend(); plt.tight_layout(); plt.savefig(FIG/'exploration.png'); plt.close()
    disagree=[s for s in S[1:] if mb.get_policy(s)!=q.get_policy(s)]
    policy=table(['状态','Part A','V','Part B'],[[s,mb.get_policy(s),f'{mb.v[s]:.4f}',q.get_policy(s)] for s in S])
    qt=table(['状态']+A,[[s]+[f'{q.q[s,a]:.4f}' for a in A] for s in S])
    perf=table(['方法/数据量','平均回报','B3 比例','B1 终止率'],[[r[0]]+[f'{v:.4f}' for v in r[1:]] for r in metrics]+[['Q/2000']+[f'{v:.4f}' for v in qm]])
    sparse=sorted(mb.counts,key=mb.counts.get)[:6]
    example=disagree[0] if disagree else 'B4'
    text=f'''# 血糖调控实验报告

## 设计与复现
运行 `python assignment3/main.py`，依赖见 requirements.txt。课程 glucose_env.py 应置于 assignment3 的上一级。算法仅使用离散经验，不使用真实奖励表、转移表或 info 中的连续血糖。训练随机动作种子 2026、环境种子 2026+i，共 2000 条轨迹，两方法使用完全相同的数据。评估每个策略使用 500 个独立环境种子 100000+i。在线实验每种模式训练 2000 回合、5 个种子。图表及本文由 main.py 生成。

## Part A
用显式计数归一化估计 P，奖励取样本均值。未访问项使用均匀分布和零奖励，这是无数据时的对称默认值，并非可信先验。价值迭代采用同步扫描，gamma=0.9，阈值 1e-4，上限 1000 次，本次迭代 {mb.iterations} 次。进入 B1 不再累积未来价值，但 reset 得到 B1 时仍可选动作，因此保留其动作估计。200 步上限作为评估截断；Part A 规划为平稳折扣模型，不建模剩余时间。动作平局按 w,g,i1,i2 处理。

![访问次数](figures/visits.png)

最稀疏的六项为 {[(s,a,mb.counts[s,a]) for s,a in sparse]}。这些项的经验模型方差更大，零访问的默认分布还可能产生乐观价值。

{policy}

状态 {example} 的 Part A 动作价值为 {[(a,round(mb._q(example,a),4)) for a in A]}，因此选择 {mb.get_policy(example)}。这个选择权衡即时奖励、动作成本和进入危险状态后失去未来收益的影响。

## 评估与消融
平均回报为无折扣累计奖励；B3 比例为所有评估动作后的 B3 次数除以总步数；B1 率为进入 B1 的回合比例，不把 reset 时的 B1 算作失败。

{perf}

![消融](figures/ablation.png)

本次三个数据量的评估指标相同，说明额外数据没有改变这些评估轨迹上的行为；这不代表模型估计完全相同，也不能证明更少数据总是足够。数据增加通常降低模型估计方差，但有限评估与离散策略切换使性能不必单调。图中橙线是完整数据训练的 Q 基线，各消融都使用共同数据前缀及相同评估种子。

## Part B
Q 初始为零，按 episode 和转移原顺序仅更新一遍，alpha=0.5。目标为 r+gamma max Q；done=True 时仅为 r，包括环境的时间上限终止。learn 不采样动作。最终 Q 表如下。

{qt}

非终止状态策略一致率为 {(8-len(disagree))/8:.1%}，不同状态为 {disagree}。Part A 将经验整理成模型后反复规划；Part B 单遍更新受样本顺序、固定学习率及价值传播速度影响。模型误差也会通过 Bellman 更新传播，因此不能仅凭方法名称断言哪条路线更优。

## 在线探索
![探索曲线](figures/exploration.png)

固定 epsilon=0.3；衰减模式每回合结束乘 0.995，下限 0.05。曲线是 50 回合滑动平均后跨种子的均值，阴影为一个标准差。最后窗口平均回报：固定 {curves['fixed'][:,-1].mean():.3f}，衰减 {curves['decay'][:,-1].mean():.3f}。固定探索持续承担随机危险动作的代价；衰减减少后期探索成本，但可能过早偏向不准确的价值估计。这里展示实际训练回报，未知最优策略，不能把曲线直接当作精确累计 Regret。

收敛要求充分访问所有状态动作、满足随机逼近条件的递减学习率，以及 gamma<1 和有界奖励。本任务满足折扣与奖励条件，但有限数据不能保证充分访问，固定 alpha=0.5 不满足通常的渐近收敛学习率条件。epsilon 下限也不能独自保证罕见状态充分访问，所以不能声称已经收敛到最优 Q。

## 讨论与反思
实现中需要特别处理终止与时间截断的不同语义，并避免离线学习中再次采样动作。小型离散空间中学模型便于重复规划和检查数据覆盖；直接学 Q 省去显式转移模型，但单遍学习可能不足。当前比较只使用一份离线训练集，评估种子相同不代表完整统计显著性检验。后续可增加训练种子、置信区间、递减学习率，或研究函数逼近。以上是实现层面的技术反思；个人学习经历应由提交者自行补充。

## 资料与工具说明
依据课程 Assignment 3 讲义和提供的模拟器接口。实现与报告初稿由 Codex 辅助生成，实验数值由本地程序运行产生；提交者应理解并核对课程对 AI 辅助的规定。
'''
    (ROOT/'report.md').write_text(text,encoding='utf-8')
    (ROOT/'analysis.md').write_text(f'''# 简答

1. 稀疏访问会同时影响两条路线，不能脱离实际数据断言绝对优劣。Part A 的少量观测可能造成转移模型偏差，并在反复规划中传播。Part B 在本作业单遍更新下还受到更新次数不足和顺序的影响，稀疏项可能保持接近零。应结合访问图、消融和评估结果判断，而非只比较最终回报。

2. 本次非终止状态的策略差异为 {disagree}。以 {example} 为例，Part A 选择 {mb.get_policy(example)}，Part B 选择 {q.get_policy(example)}。可能原因是经验模型误差和直接价值更新的传播路径不同，固定学习率还使 Q 更受后期样本影响。这是可能解释，不能据此认定某种方法获得了真实最优动作。
''',encoding='utf-8')
    print(perf)

if __name__=='__main__': main()

