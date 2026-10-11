"""Validate the completed experiment and append its evidence without editing old text."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

folder = Path(__file__).resolve().parent
project = folder.parent
rows = list(csv.DictReader((folder / 'matches.csv').open(encoding='utf-8-sig')))
assert len(rows) == 112, f'Expected 112 completed matches, got {len(rows)}'
meta = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
for name, digest in meta['hashes'].items():
    assert hashlib.sha256((project / name).read_bytes()).hexdigest() == digest, name
before = (folder / 'report_before.md').read_bytes()
assert (project / 'report.md').read_bytes() == before, 'Report changed during the run; do not overwrite user edits'

def subset(n, kind, color=None):
    return [r for r in rows if int(r['board']) == n and r['kind'] == kind
            and (color is None or int(r['a_color']) == color)]

def count(group):
    return '/'.join(str(sum(r['outcome'] == s for r in group)) for s in ('win', 'loss', 'draw'))

def average(group, both=False):
    total = sum(float(r['a_total_seconds']) for r in group)
    moves = sum(int(r['a_moves']) for r in group)
    if both:
        total += sum(float(r['b_total_seconds']) for r in group)
        moves += sum(int(r['b_moves']) for r in group)
    return total / moves

for n in (5, 9):
    for kind, total in (('random', 40), ('self', 8), ('depth1', 8)):
        group = subset(n, kind)
        assert len(group) == total
        for color in (1, 2):
            assert len(subset(n, kind, color)) == total // 2
        assert all(float(r['limit']) == 5 for r in group)
        for r in group:
            if not r['failure']:
                assert int(r['a_moves']) + int(r['b_moves']) == int(r['moves'])
                assert max(float(r['a_max_seconds']), float(r['b_max_seconds'])) <= 5

lines = ['\n\n## 6. 补充实验评估：增强版 AI（2026-10-11）', '',
         '本节为新增内容，前文与旧实验数据保持原样。测试对象是加入活三/跳三、双三、四三、双四识别以及紧急搜索延伸后的当前 AI。', '',
         '### 6.1 实验设置与复现', '',
         f"运行环境：Python {meta['python'].split()[0]}，{meta['platform']}。",
         '每步时限为作业默认的 **5.0 秒**，两组配置为 **(N,K)=(5,3)、(9,4)**。每组随机对战 40 局、增强版自我对战 8 局、常规搜索深度对比 8 局，共 **112 局**；各组均交换先后手。',
         '随机种子从 20261008 起逐局递增。每局重新创建 AI，计时与规则使用项目的 `timed_move` 和 `Game`，并非本次逐局调用课程 `arena.py`。', '',
         '```powershell', meta['command'], '```', '',
         '统计口径：胜率=胜局/总局数，和棋不算获胜；平均每步耗时=累计调用时间/调用次数，包含线程开销；局长为双方合计的成功落子数。耗时按所有落子加权，不是各局平均值的简单平均。',
         '自我对战表中的耗时统计双方；随机和深度对比表中的耗时统计增强版 A。异常调用也计入调用时间，失败局保留在胜负统计中。', '',
         '### 6.2 对随机 AI 的胜率与耗时', '',
         '| N,K | A 执棋 | 局数 | 胜/负/和 | 胜率 | A 平均每步（秒） | A 最慢一步（秒） |',
         '|---|---|---:|---|---:|---:|---:|']
for n, k in ((5, 3), (9, 4)):
    for color, label in ((1, '黑棋'), (2, '白棋'), (None, '合计')):
        g = subset(n, 'random', color)
        wins = sum(r['outcome'] == 'win' for r in g)
        lines.append(f"| {n},{k} | {label} | {len(g)} | {count(g)} | {wins/len(g):.1%} | {average(g):.4f} | {max(float(r['a_max_seconds']) for r in g):.4f} |")
lines += ['', '每组先后手各 20 局，满足随机基线对战样本要求。', '',
          '### 6.3 增强版自我对战', '',
          '| N,K | 局数 | 黑胜/白胜/和 | A 执黑胜/负/和 | A 执白胜/负/和 | 平均局长（步） | 双方平均每步（秒） |',
          '|---|---:|---|---|---|---:|---:|']
for n, k in ((5, 3), (9, 4)):
    g = subset(n, 'self')
    wins = '/'.join(str(sum(int(r['winner']) == p for r in g)) for p in (1, 2, 0))
    lines.append(f"| {n},{k} | {len(g)} | {wins} | {count(subset(n,'self',1))} | {count(subset(n,'self',2))} | {statistics.mean(int(r['moves']) for r in g):.2f} | {average(g,True):.4f} |")
lines += ['', 'A/B 为统计标签，算法相同；A 执黑、执白各 4 局。同一算法接近确定性，交换标签不等于独立随机样本，不能据此推导理论先手必胜或显著性结论。', '',
          '### 6.4 不同常规搜索深度对比', '',
          'A 使用增强版默认迭代加深（常规深度上限 64，实际受时间限制）；B 使用相同增强版、仅将 `max_depth=1`。双方时间预算、估值和威胁规则相同。',
          '**B 不是严格只看一步**：直接战术判断和最多额外 4 层的紧急延伸仍然启用。本实验比较“常规深度上限 1”与“默认迭代加深”，不是移除增强功能的消融实验。', '',
          '| N,K | 局数 | A 胜/负/和 | A 胜率 | A 执黑胜/负/和 | A 执白胜/负/和 | A 平均每步（秒） | 平均局长（步） |',
          '|---|---:|---|---:|---|---|---:|---:|']
for n, k in ((5, 3), (9, 4)):
    g = subset(n, 'depth1')
    wins = sum(r['outcome'] == 'win' for r in g)
    lines.append(f"| {n},{k} | {len(g)} | {count(g)} | {wins/len(g):.1%} | {count(subset(n,'depth1',1))} | {count(subset(n,'depth1',2))} | {average(g):.4f} | {statistics.mean(int(r['moves']) for r in g):.2f} |")
lines += ['', '### 6.5 配置对比与讨论', '']
for n, k in ((5, 3), (9, 4)):
    g = subset(n, 'random')
    wins = sum(r['outcome'] == 'win' for r in g)
    lines.append(f"- ({n},{k}) 对随机 AI：{wins}/{len(g)} 胜，平均每步 {average(g):.4f} 秒，平均 {statistics.mean(int(r['moves']) for r in g):.2f} 步结束；自我对战平均 {statistics.mean(int(r['moves']) for r in subset(n,'self')):.2f} 步。")
lines += ['', '不同 N 与 K 同时改变，棋盘规模、连珠目标、局长和搜索成本都会受到影响，因此不能将差异单独归因于棋盘大小或增强策略。',
          '与前文旧版实验相比，本次时限由 0.05 秒变为 5 秒；即使使用同样的初始随机种子，AI 落子也会改变后续随机可选位置，因此这不是严格控制变量的新旧版本对比，不能把胜率差异全部解释为增强效果。',
          '随机对手衡量基础能力，自我对战和深度对比每组仅 8 局，不能充分代表面对人类或其他强 AI 的表现；本次未在标准 (15,5) 上进行大样本实验，也未单独隔离活三/双威胁模块的贡献。', '',
          '### 6.6 合规性与原始证据', '']
failures = [r for r in rows if r['failure']]
lines.append(f"112 局中记录到 **{len(failures)} 局协议失败**（超时、异常或非法落子）；所有调用中最大记录耗时为 {max(max(float(r['a_max_seconds']),float(r['b_max_seconds'])) for r in rows):.4f} 秒。")
for r in failures:
    lines.append(f"- ({r['board']},{r['win']}) {r['kind']} 第 {r['game']} 局：{r['failure']}")
lines += ['', '- 逐局原始数据：[matches.csv](results_enhanced_20261011/matches.csv)。',
          '- 运行日志：[run.log](results_enhanced_20261011/run.log)。',
          '- 参数、运行环境和文件哈希：[metadata.json](results_enhanced_20261011/metadata.json)。',
          '- 被测 AI 快照：[gomoku_ai_snapshot.py](results_enhanced_20261011/gomoku_ai_snapshot.py)。',
          f"- 被测 AI SHA-256：`{meta['hashes']['gomoku_ai.py']}`。", '',
          '墙钟时间受系统调度影响；种子与代码快照用于提高可追溯性，不保证每次运行的搜索深度和胜负完全相同。', '']
addition = '\n'.join(lines).encode('utf-8')
(folder / 'report_supplement.md').write_bytes(addition)
with (project / 'report.md').open('ab') as stream:
    stream.write(addition)
assert (project / 'report.md').read_bytes() == before + addition
print(f'Appended supplement; verified {len(before)} original bytes unchanged; {len(rows)} matches, {len(failures)} failures.')
