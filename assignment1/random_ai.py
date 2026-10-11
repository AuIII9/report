"""Random baseline, adapted from the course-provided random_ai.py."""
import random

NAME = "RandomBaseline"


class GomokuAI:
    def __init__(self, player_id, board_size, win_length):
        self.player_id = player_id
        self.board_size = board_size
        self.win_length = win_length

    def get_move(self, board, last_opponent_move, time_limit):
        return random.choice([(r, c) for r, row in enumerate(board)
                              for c, v in enumerate(row) if v == 0])
