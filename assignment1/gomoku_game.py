"""Game rules, external protocol loading, and timed calls shared by both UIs."""
import importlib.util
import itertools
from pathlib import Path
import queue
import sys
import threading
import time

_ids = itertools.count()


class Game:
    def __init__(self, size=15, win=5):
        if not 3 <= win <= size:
            raise ValueError("必须满足 3 ≤ 连珠数 ≤ 棋盘大小")
        self.size, self.win = size, win
        self.board = [[0] * size for _ in range(size)]
        self.player = 1
        self.winner = None  # None: ongoing; 0: draw; 1/2: winner
        self.history = []

    def validate(self, move):
        if not isinstance(move, tuple) or len(move) != 2:
            raise ValueError("落子必须是 (row, col) 元组")
        r, c = move
        if type(r) is not int or type(c) is not int:
            raise ValueError("坐标必须为整数")
        if not (0 <= r < self.size and 0 <= c < self.size):
            raise ValueError("坐标超出棋盘")
        if self.board[r][c]:
            raise ValueError("该位置已有棋子")

    def play(self, move):
        if self.winner is not None:
            raise ValueError("本局已经结束")
        self.validate(move)
        r, c = move
        p = self.player
        self.board[r][c] = p
        self.history.append((r, c))
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            count = 1
            for sign in (-1, 1):
                rr, cc = r + dr * sign, c + dc * sign
                while 0 <= rr < self.size and 0 <= cc < self.size and self.board[rr][cc] == p:
                    count += 1
                    rr, cc = rr + dr * sign, cc + dc * sign
            if count >= self.win:
                self.winner = p
                return
        if len(self.history) == self.size ** 2:
            self.winner = 0
        else:
            self.player = 3 - p

    def text(self):
        return "    " + " ".join(f"{c:2}" for c in range(self.size)) + "\n" + "\n".join(
            f"{r:2}  " + " ".join(f"{'.XO'[v]:>2}" for v in row)
            for r, row in enumerate(self.board))


def load_ai(path):
    if path == "builtin":
        from gomoku_ai import GomokuAI
        return GomokuAI
    if path == "random":
        from random_ai import GomokuAI
        return GomokuAI
    source = Path(path).resolve()
    if not source.is_file():
        raise ValueError(f"AI 文件不存在：{source}")
    name = f"gomoku_external_{next(_ids)}"
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ValueError("请选择 Python AI 文件")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.GomokuAI


def timed_move(ai, board, last_move, limit):
    """Return (move, seconds, error). Timed-out daemon threads are discarded."""
    result = queue.Queue(maxsize=1)
    snapshot = [row[:] for row in board]

    def work():
        try:
            result.put((ai.get_move(snapshot, last_move, limit), None))
        except BaseException as exc:
            result.put((None, f"AI 异常：{exc!r}"))

    start = time.perf_counter()
    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    worker.join(limit)
    elapsed = time.perf_counter() - start
    if worker.is_alive() or elapsed > limit:
        return None, elapsed, "AI 超时"
    move, error = result.get_nowait()
    return move, elapsed, error
