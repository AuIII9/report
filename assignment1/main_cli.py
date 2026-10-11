"""Command-line game: human/human, human/AI, AI/AI."""
import argparse
import math
from gomoku_game import Game, load_ai, timed_move


def main(argv=None):
    parser = argparse.ArgumentParser(description="广义五子棋：坐标从 0 开始，输入 q 退出")
    parser.add_argument("--mode", choices=("hvh", "hva", "ava"), default="hva")
    parser.add_argument("--board", type=int, default=15)
    parser.add_argument("--win", type=int, default=5)
    parser.add_argument("--time", type=float, default=5.0)
    parser.add_argument("--human", type=int, choices=(1, 2), default=1)
    parser.add_argument("--ai-black", default="builtin", help="builtin / random / AI 文件路径")
    parser.add_argument("--ai-white", default="builtin")
    args = parser.parse_args(argv)
    if not math.isfinite(args.time) or args.time <= 0:
        parser.error("--time 必须大于 0")
    try:
        game = Game(args.board, args.win)
        ais = {}
        for p, path in ((1, args.ai_black), (2, args.ai_white)):
            if args.mode == "ava" or (args.mode == "hva" and p != args.human):
                ais[p] = load_ai(path)(p, args.board, args.win)
    except Exception as exc:
        parser.error(str(exc))
    print("黑棋 X 先行，白棋 O 后行。输入：行 列，例如 7 7；q 退出。")
    print(game.text())
    while game.winner is None:
        p = game.player
        label = "黑棋" if p == 1 else "白棋"
        if p in ais:
            move, elapsed, error = timed_move(ais[p], game.board,
                                              game.history[-1] if game.history else None, args.time)
            try:
                if error:
                    raise ValueError(error)
                game.play(move)
            except ValueError as exc:
                print(f"{label}判负：{exc}。{'白棋' if p == 1 else '黑棋'}获胜。")
                return
            print(f"{label} AI 落子 {move}，耗时 {elapsed:.3f} 秒")
        else:
            try:
                raw = input(f"{label}落子 > ").strip()
                if raw.lower() in ("q", "quit", "exit"):
                    return
                move = tuple(int(v) for v in raw.replace(",", " ").split())
                game.play(move)
            except (EOFError, KeyboardInterrupt):
                print("\n已退出。")
                return
            except ValueError as exc:
                print(f"输入无效：{exc}")
                continue
        print(game.text())
    print("和棋！" if game.winner == 0 else f"{'黑棋' if game.winner == 1 else '白棋'}获胜！")


if __name__ == "__main__":
    main()
