# 广义五子棋

Python 3.10+，仅使用标准库。Windows 官方 Python 一般附带 tkinter，无需 pip 安装。
所有下列命令均在本项目根目录（`assignment1` 的上一级）运行。

## 启动

图形界面：

Windows 也可以双击本目录的 `launch.bat` 启动。

```powershell
python assignment1/main.py
```

选择人人/人机/机机模式，设置 N、K、时限后点击开始。人机可选择执黑或执白。
黑白两方的 AI 输入框可填 `builtin`、`random` 或符合统一协议的 Python 文件路径。
点击棋盘交叉点落子，红点标记上一步；修改设置后重新开始才会应用。

命令行界面（输入 `行 列`，从 0 开始；输入 `q` 退出）：

```powershell
python assignment1/main.py --cli --mode hvh --board 9 --win 4
python assignment1/main.py --cli --mode hva --human 1 --board 15 --win 5 --time 5
python assignment1/main.py --cli --mode ava --ai-black builtin --ai-white random --board 9 --win 4 --time 1
```

`--human 2` 表示人类执白。`--ai-black` 和 `--ai-white` 均支持外部 `.py` 文件路径。
也可以直接运行 `python assignment1/main_cli.py ...`。
GUI 将 N 限制为课程常用的 5–15；CLI、规则引擎和 AI 使用构造参数，支持更大棋盘。

## 使用老师的裁判

```powershell
python assignment-1-code/arena.py assignment1/gomoku_ai.py assignment-1-code/random_ai.py --board 15 --win 5 --time 5 --games 2 --quiet
python assignment-1-code/arena.py assignment1/gomoku_ai.py assignment-1-code/random_ai.py --board 9 --win 4 --time 1 --games 20 --quiet
```

老师的原始文件保持不变。`gomoku_ai.py` 可单独复制并被裁判加载，无本地模块或第三方依赖。
自己的游戏按协议严格要求返回整数元组；外部 AI 的超时、异常、非法落子会判负。
与课程裁判一样，超时采用守护线程判定，无法强制停止已超时的 Python 线程；不应用于隔离恶意代码。

## 测试与实验

```powershell
python -m unittest discover -s assignment1 -v
python assignment1/experiments.py --time 0.1
```

实验默认使用 (5,3)、(9,4) 两组配置，每组随机对战 40 局（先后手各 20 局）、自我对战 8 局、与一层搜索对战 8 局。
生成 `results/matches.csv` 和 `results/summary.md`。短时限方便复现；正式评估可以改为 `--time 5`，耗时会明显增加。
实验仅在结束后写出结果，重新运行会覆盖该目录下的同名实验输出；可用 `--output` 指定新目录。

## 文件说明

| 文件 | 作用 |
|---|---|
| `gomoku_ai.py` | 统一协议 AI：Minimax/Negamax、Alpha-Beta、迭代加深 |
| `gomoku_game.py` | 棋盘状态、规则判定、外部 AI 加载、计时 |
| `gui.py` | Tk 图形界面 |
| `main.py` / `main_cli.py` | 图形/命令行入口 |
| `random_ai.py` | 基于课程示例的随机对手 |
| `experiments.py` | 实验执行和统计 |
| `test_gomoku.py` | 规则、战术、时限与协议测试 |
| `report.md` | 设计说明和实验报告草稿 |

## 作业使用说明

### 2026-10-09 战术防守更新

AI 新增活三、跳三、四三、双三及双四威胁识别，按不同获胜位置去重，避免把重叠窗口误判为多路必胜。
优先级仍为直接获胜、应对对手直接获胜；没有必须立即防守的位置时，也会主动制造自身的双获胜点。
面对对方下一步可以制造双获胜点的局面，优先保留能消除全部此类威胁的防守，或能够抢先形成冲四的强制进攻。
在搜索深度边界，对冲四或双获胜点威胁最多额外搜索 4 层。双三、四三作为强威胁评分，不直接认定必胜。
这些规则随 K 自适应。战术回归测试覆盖活三、跳三、双三、四三、双四、先取胜再防守、边界和不同 K。
`results/` 中原有 112 局实验数据来自更新前版本，不能作为新版胜率；要评估新版请重新运行实验并使用新的 `--output` 目录。

课程要求独立完成。本实现及报告草稿由 AI 辅助生成，请按照课程对 AI 辅助的规定使用，理解代码后自行修改、复现实验并如实披露来源，不要将自动生成内容冒充独立完成的工作。
发布到课程仓库前检查 `assignment1/gomoku_ai.py` 和 `assignment1/report.md` 路径。此项目没有自动提交或推送 GitHub。
