"""Reproducible matches. Generates raw CSV and a Markdown results section."""
import argparse
import csv
import math
from pathlib import Path
import platform
import random
import statistics
import sys
from gomoku_ai import GomokuAI
from gomoku_game import Game, timed_move
from random_ai import GomokuAI as RandomAI


def match(n, k, seconds, number, kind, seed):
    random.seed(seed)
    a_color = 1 if number % 2 == 0 else 2
    b_color = 3 - a_color
    a = GomokuAI(a_color, n, k)
    b = RandomAI(b_color, n, k) if kind == "random" else GomokuAI(b_color, n, k)
    if kind == "depth1":
        b.max_depth = 1
    players = {a_color: a, b_color: b}
    game = Game(n, k)
    times = {1: [], 2: []}
    depths = []
    failure = ""
    while game.winner is None:
        p = game.player
        move, elapsed, error = timed_move(players[p], game.board,
                                          game.history[-1] if game.history else None, seconds)
        times[p].append(elapsed)
        if p == a_color:
            depths.append(a.last_depth)
        try:
            if error:
                raise ValueError(error)
            game.play(move)
        except ValueError as exc:
            failure = str(exc)
            game.winner = 3 - p
    return dict(kind=kind, board=n, win=k, game=number + 1, seed=seed,
                limit=seconds, a_color=a_color, winner=game.winner,
                outcome="draw" if game.winner == 0 else "win" if game.winner == a_color else "loss",
                moves=len(game.history), a_moves=len(times[a_color]),
                a_total_seconds=sum(times[a_color]),
                a_mean_seconds=statistics.mean(times[a_color]) if times[a_color] else 0,
                a_max_seconds=max(times[a_color], default=0),
                b_moves=len(times[b_color]), b_total_seconds=sum(times[b_color]),
                b_max_seconds=max(times[b_color], default=0),
                a_mean_depth=statistics.mean(depths) if depths else 0, failure=failure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time", type=float, default=0.1)
    parser.add_argument("--games", type=int, default=40, help="random matches per configuration (even, >=40)")
    parser.add_argument("--self-games", type=int, default=8)
    parser.add_argument("--depth-games", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    if args.games < 40 or args.games % 2 or not math.isfinite(args.time) or args.time <= 0 or args.self_games < 2 or args.depth_games < 2:
        parser.error("需要偶数且至少 40 局随机对战，正时限，自我/深度对战至少 2 局")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for n, k in ((5, 3), (9, 4)):
        for kind, total in (("random", args.games), ("self", args.self_games), ("depth1", args.depth_games)):
            for i in range(total):
                row = match(n, k, args.time, i, kind, args.seed + len(rows))
                rows.append(row)
                # Persist each completed game so longer 5-second runs retain evidence.
                with (args.output / "matches.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"{n}x{n} K={k} {kind} {i + 1}/{total}: {row['outcome']} moves={row['moves']} {row['failure']}", flush=True)
    with (args.output / "matches.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["## 实际实验结果", "", f"环境：Python {sys.version.split()[0]}；{platform.platform()}。",
             f"每步时限 {args.time} 秒；随机种子从 {args.seed} 起逐局递增。",
             "线程调度会影响搜索深度，因此同一随机种子也不保证结果完全一致。" + ("本次为短时限实验，不代表默认 5 秒时限性能。" if args.time < 5 else "本次使用每步 5 秒时限。"), "",
             "A 为默认迭代加深 AI；random 为随机对手，self 为相同 AI，depth1 将常规搜索上限设为一层，但仍保留战术判断与最多四层的紧急延伸。每组交替先后手。",
             "平均耗时以 A 的所有实际落子调用加权计算，包含计时线程开销；胜率为胜局数除以总局数。", "",
             "| N,K | 对手 | 局数 | A 胜/负/和 | A 胜率 | A 平均每步秒 | 平均局长 | 判负异常数 |",
             "|---|---|---:|---|---:|---:|---:|---:|"]
    for n, k in ((5, 3), (9, 4)):
        for kind in ("random", "self", "depth1"):
            group = [r for r in rows if r["board"] == n and r["kind"] == kind]
            wins, losses, draws = (sum(r["outcome"] == o for r in group) for o in ("win", "loss", "draw"))
            mean_time = sum(r["a_total_seconds"] for r in group) / sum(r["a_moves"] for r in group)
            lines.append(f"| {n},{k} | {kind} | {len(group)} | {wins}/{losses}/{draws} | {wins/len(group):.1%} | {mean_time:.4f} | {statistics.mean(r['moves'] for r in group):.2f} | {sum(bool(r['failure']) for r in group)} |")
    lines.extend(["", "随机对战按 A 的执棋颜色细分：", "",
                  "| N,K | A 执棋 | 局数 | 胜/负/和 |", "|---|---|---:|---|"])
    for n in (5, 9):
        for color in (1, 2):
            group = [r for r in rows if r["board"] == n and r["kind"] == "random" and r["a_color"] == color]
            counts = "/".join(str(sum(r["outcome"] == o for r in group)) for o in ("win", "loss", "draw"))
            lines.append(f"| {n},{3 if n == 5 else 4} | {'黑' if color == 1 else '白'} | {len(group)} | {counts} |")
    lines.extend(["", "原始逐局数据见同目录的 `matches.csv`。对局数较少的自我对战和深度对比仅供观察，不能据此认定某算法具有稳定优势。", ""])
    (args.output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
