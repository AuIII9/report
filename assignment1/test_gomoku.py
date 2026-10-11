"""Run with: python -m unittest discover -s assignment1 -v"""
import copy
import time
import unittest
from gomoku_game import Game, timed_move
from gomoku_ai import GomokuAI


class RuleTests(unittest.TestCase):
    def test_all_directions_and_long_runs(self):
        for dr, dc, r, c in ((0, 1, 2, 0), (1, 0, 0, 2), (1, 1, 0, 0), (1, -1, 0, 4)):
            for k in (3, 4, 5):
                game = Game(5, k)
                for i in range(k):
                    if i != 1:
                        game.board[r + dr * i][c + dc * i] = 1
                game.play((r + dr, c + dc))
                self.assertEqual(game.winner, 1)
        game = Game(7, 4)
        for c in (0, 1, 3, 4):
            game.board[2][c] = 1
        game.play((2, 2))
        self.assertEqual(game.winner, 1)

    def test_draw_and_invalid(self):
        game = Game(3, 3)
        for move in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 0), (1, 2), (2, 1), (2, 0), (2, 2)):
            game.play(move)
        self.assertEqual(game.winner, 0)
        game = Game(5, 3)
        for move in ((-1, 0), (5, 0), (True, 0), [0, 0], (0,)):
            with self.assertRaises(ValueError):
                game.play(move)
        game.play((0, 0))
        with self.assertRaises(ValueError):
            game.play((0, 0))


class AITests(unittest.TestCase):
    def test_open_and_broken_three_defence(self):
        # An unrelated own two must not distract from an opponent's live three.
        for stones, allowed in (
            ([(7, 6), (7, 7), (7, 8)], {(7, 5), (7, 9)}),
            ([(7, 5), (7, 6), (7, 8)], {(7, 7)}),
        ):
            for transpose in (False, True):
                for player in (1, 2):
                    board = [[0] * 15 for _ in range(15)]
                    transform = lambda m: m[::-1] if transpose else m
                    for cell in stones:
                        r, c = transform(cell)
                        board[r][c] = 3 - player
                    board[2][2] = board[2][3] = player
                    before = copy.deepcopy(board)
                    move = GomokuAI(player, 15, 5).get_move(board, None, 0.2)
                    self.assertIn(move, {transform(m) for m in allowed})
                    self.assertEqual(board, before)

    def test_prevent_multi_line_threats(self):
        for enemy, own in (
            ([(7, 5), (7, 6), (5, 7), (6, 7)], [(2, 2), (2, 3)]),
            ([(7, 4), (7, 5), (7, 6), (5, 7), (6, 7)], [(7, 3), (2, 2)]),
            ([(7, 4), (7, 5), (7, 6), (4, 7), (5, 7), (6, 7)], [(7, 3), (3, 7)]),
        ):
            board = [[0] * 15 for _ in range(15)]
            for r, c in enemy:
                board[r][c] = 2
            for r, c in own:
                board[r][c] = 1
            self.assertEqual(GomokuAI(1, 15, 5).get_move(board, None, 0.3), (7, 7))

    def test_win_before_defence_and_create_own_fork(self):
        board = [[0] * 15 for _ in range(15)]
        for c in (5, 6, 7):
            board[7][c] = 2
        for c in (1, 2, 3, 4):
            board[1][c] = 1
        self.assertIn(GomokuAI(1, 15, 5).get_move(board, None, 0.2), {(1, 0), (1, 5)})
        board[1][4] = 0
        # Both players have a live three: ours moves first and makes open four.
        self.assertEqual(GomokuAI(1, 15, 5).get_move(board, None, 0.2), (1, 4))

    def test_threats_adapt_to_k_and_edges(self):
        for k in (4, 5, 6, 7):
            board = [[0] * 15 for _ in range(15)]
            for c in range(4, 4 + k - 2):
                board[7][c] = 2
            self.assertIn(GomokuAI(1, 15, k).get_move(board, None, 0.2), {(7, 3), (7, k + 2)})
        ai = GomokuAI(1, 15, 5)
        ai.deadline = time.perf_counter() + 1
        board = [[0] * 15 for _ in range(15)]
        for c in (0, 1, 2):
            board[0][c] = 2
        self.assertEqual(ai._tactics(board, 2)[1][(0, 3)], {(0, 4)})
        board[7][4] = 1
        for c in (5, 6, 7):
            board[7][c] = 2
        # Overlapping windows must not count a single winning square twice.
        self.assertEqual(ai._tactics(board, 2)[1][(7, 8)], {(7, 9)})

    def test_immediate_win_and_block_multiple_k(self):
        for n, k in ((5, 3), (9, 4), (15, 5), (9, 7)):
            for p in (1, 2):
                board = [[0] * n for _ in range(n)]
                for c in range(k - 1):
                    board[0][c] = p
                for ai_player in (p, 3 - p):
                    ai = GomokuAI(ai_player, n, k)
                    self.assertEqual(ai.get_move(board, None, 0.2), (0, k - 1))

    def test_both_diagonals_tactics(self):
        for cells, target in (([(0, 0), (1, 1), (2, 2)], (3, 3)),
                              ([(0, 4), (1, 3), (2, 2)], (3, 1))):
            board = [[0] * 5 for _ in range(5)]
            for r, c in cells:
                board[r][c] = 2
            self.assertEqual(GomokuAI(1, 5, 4).get_move(board, None, 0.2), target)

    def test_time_and_no_mutation(self):
        board = [[0] * 15 for _ in range(15)]
        board[7][7], board[7][8], board[8][7] = 1, 2, 1
        before = copy.deepcopy(board)
        ai = GomokuAI(2, 15, 5)
        for limit in (0.001, 0.05, 0.2):
            start = time.perf_counter()
            move = ai.get_move(board, (8, 7), limit)
            elapsed = time.perf_counter() - start
            self.assertEqual(board, before)
            self.assertIsInstance(move, tuple)
            self.assertEqual(board[move[0]][move[1]], 0)
            self.assertLess(elapsed, limit + 0.03)

    def test_timeout_and_exception_adjudication(self):
        class Slow:
            def get_move(self, *args):
                time.sleep(0.1)
                return (0, 0)
        class Broken:
            def get_move(self, *args):
                raise RuntimeError("test")
        self.assertEqual(timed_move(Slow(), [[0]], None, 0.01)[2], "AI 超时")
        self.assertIn("AI 异常", timed_move(Broken(), [[0]], None, 0.1)[2])

    def test_all_assignment_configurations(self):
        for n in range(5, 16):
            for k in range(3, n + 1):
                game = Game(n, k)
                game.play((n // 2, n // 2))
                move = GomokuAI(2, n, k).get_move(game.board, game.history[-1], 0.006)
                game.validate(move)


if __name__ == "__main__":
    unittest.main()
