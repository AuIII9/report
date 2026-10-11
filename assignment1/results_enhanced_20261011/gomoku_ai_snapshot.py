"""Parameterised Gomoku AI. Standard library only; protocol v1.

Negamax is the zero-sum formulation of Minimax. Iterative deepening keeps
the last fully completed answer when the wall-clock budget expires.
"""
import time

NAME = "AdaptiveSearch"
DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))
MATE = 10 ** 12


class _Deadline(Exception):
    pass


class GomokuAI:
    def __init__(self, player_id: int, board_size: int, win_length: int):
        if player_id not in (1, 2) or not 3 <= win_length <= board_size:
            raise ValueError("Invalid player, board size or win length")
        self.player_id = player_id
        self.board_size = board_size
        self.win_length = win_length
        self.max_depth = 64
        self.last_depth = 0
        self.nodes = 0
        self._tactic_cache = {}
        n, k = board_size, win_length
        self.windows = []
        for r in range(n):
            for c in range(n):
                for dr, dc in DIRS:
                    if 0 <= r + (k - 1) * dr < n and 0 <= c + (k - 1) * dc < n:
                        self.windows.append(tuple((r + i * dr, c + i * dc) for i in range(k)))
        # Normalised by K: threat weights remain bounded even on larger boards.
        self.weights = [0] + [10 ** (6 * i / k) for i in range(1, k + 1)]

    def _check_time(self):
        if time.perf_counter() >= self.deadline:
            raise _Deadline

    def _wins(self, b, r, c, p):
        n, k = self.board_size, self.win_length
        for dr, dc in DIRS:
            count = 1
            for sign in (-1, 1):
                rr, cc = r + sign * dr, c + sign * dc
                while 0 <= rr < n and 0 <= cc < n and b[rr][cc] == p:
                    count += 1
                    rr, cc = rr + sign * dr, cc + sign * dc
            if count >= k:
                return True
        return False

    def _local_score(self, b, r, c, p):
        """Score contiguous runs formed by a hypothetical move, with open ends."""
        n, k = self.board_size, self.win_length
        value = 0.0
        for dr, dc in DIRS:
            count, opens = 1, 0
            for sign in (-1, 1):
                rr, cc = r + sign * dr, c + sign * dc
                while 0 <= rr < n and 0 <= cc < n and b[rr][cc] == p:
                    count += 1
                    rr, cc = rr + sign * dr, cc + sign * dc
                if 0 <= rr < n and 0 <= cc < n and b[rr][cc] == 0:
                    opens += 1
            if count >= k:
                return MATE
            value += self.weights[count] * (opens + 0.1)
        return value

    def _tactics(self, b, p):
        """Distinct winning squares, not counts of overlapping windows.

        pairs[m] lists winning replies created by move m. A move with two
        distinct replies is a winning fork (open four / double four for K=5).
        threes[m] contains directions in which m permits a later open four;
        this also recognises broken threes, and scales with K.
        """
        key = (p, tuple(tuple(row) for row in b))
        if key in self._tactic_cache:
            return self._tactic_cache[key]
        wins, pairs, extensions = set(), {}, {}
        for i, cells in enumerate(self.windows):
            if i % 32 == 0:
                self._check_time()
            empty = []
            for r, c in cells:
                if b[r][c] == 3 - p:
                    break
                if not b[r][c]:
                    empty.append((r, c))
                    if len(empty) > 3:
                        break
            else:
                if len(empty) == 1:
                    wins.add(empty[0])
                elif len(empty) == 2:
                    a, z = empty
                    pairs.setdefault(a, set()).add(z)
                    pairs.setdefault(z, set()).add(a)
                elif len(empty) == 3 and self.win_length >= 4:
                    direction = (cells[1][0] - cells[0][0], cells[1][1] - cells[0][1])
                    for move in empty:
                        for follow in empty:
                            if follow != move:
                                target = next(t for t in empty if t != move and t != follow)
                                extensions.setdefault((move, follow, direction), set()).add(target)
        threes = {}
        for (move, follow, direction), targets in extensions.items():
            if len(targets) >= 2:
                threes.setdefault(move, set()).add(direction)
        result = wins, pairs, threes
        if len(self._tactic_cache) >= 2048:
            self._tactic_cache.clear()
        self._tactic_cache[key] = result
        return result

    @staticmethod
    def _threat_score(move, pairs, threes):
        fours = len(pairs.get(move, ()))
        live_threes = len(threes.get(move, ()))
        if fours >= 2:
            return 100_000_000  # independent next-move wins
        if fours and live_threes:
            return 10_000_000   # four-three: forcing threat plus another line
        if live_threes >= 2:
            return 5_000_000    # double three; strong, not a proven win
        return fours * 200_000 + live_threes * 100_000

    def _candidates(self, b, p, depth, preferred=None):
        n = self.board_size
        nearby = set()
        for r in range(n):
            self._check_time()
            for c in range(n):
                if b[r][c]:
                    for rr in range(max(0, r - 2), min(n, r + 3)):
                        for cc in range(max(0, c - 2), min(n, c + 3)):
                            if b[rr][cc] == 0:
                                nearby.add((rr, cc))
        if not nearby:
            return [(n // 2, n // 2)] if b[n // 2][n // 2] == 0 else []
        own_wins, own_pairs, own_threes = self._tactics(b, p)
        enemy_wins, enemy_pairs, enemy_threes = self._tactics(b, 3 - p)
        if own_wins:
            return sorted(own_wins)
        if enemy_wins:
            return sorted(enemy_wins)
        ranked, urgent = [], set()
        for r, c in sorted(nearby):
            self._check_time()
            attack = self._local_score(b, r, c, p)
            defend = self._local_score(b, r, c, 3 - p)
            move = (r, c)
            attack_threat = self._threat_score(move, own_pairs, own_threes)
            defend_threat = self._threat_score(move, enemy_pairs, enemy_threes)
            if max(attack_threat, defend_threat) >= 5_000_000:
                urgent.add(move)
            central = abs(r - (n - 1) / 2) + abs(c - (n - 1) / 2)
            ranked.append((attack_threat + defend_threat * 1.2 + attack + defend * 1.15 - central * 0.01, move))
        ranked.sort(reverse=True)
        moves = [move for _, move in ranked]
        if preferred in moves:
            moves.remove(preferred)
            moves.insert(0, preferred)
        # Selective search trades exhaustive coverage for useful search depth.
        selected = moves[:18 if depth <= 2 else 12]
        return selected + [m for m in moves if m in urgent and m not in selected]

    def _evaluate(self, b, p):
        score = 0.0
        for i, cells in enumerate(self.windows):
            if i % 64 == 0:
                self._check_time()
            a = z = 0
            for r, c in cells:
                if b[r][c] == p:
                    a += 1
                elif b[r][c] == 3 - p:
                    z += 1
            if a and not z:
                score += self.weights[a]
            elif z and not a:
                score -= self.weights[z]
        _, own_pairs, own_threes = self._tactics(b, p)
        _, enemy_pairs, enemy_threes = self._tactics(b, 3 - p)
        own = max((self._threat_score(m, own_pairs, own_threes)
                   for m in own_pairs.keys() | own_threes.keys()), default=0)
        enemy = max((self._threat_score(m, enemy_pairs, enemy_threes)
                     for m in enemy_pairs.keys() | enemy_threes.keys()), default=0)
        return score + own - enemy

    def _search(self, b, p, depth, alpha, beta, empty_count, ply, extensions=4):
        self._check_time()
        self.nodes += 1
        if empty_count == 0:
            return 0
        wins, pairs, _ = self._tactics(b, p)
        enemy_wins, enemy_pairs, _ = self._tactics(b, 3 - p)
        if wins:
            return MATE - ply
        if len(enemy_wins) >= 2:
            return -MATE + ply + 1
        if not enemy_wins and any(len(points) >= 2 for points in pairs.values()):
            return MATE - ply - 2
        if depth == 0:
            if extensions and (enemy_wins or any(len(points) >= 2 for points in enemy_pairs.values())):
                depth, extensions = 1, extensions - 1
            else:
                return self._evaluate(b, p)
        value = -float("inf")
        for r, c in self._candidates(b, p, depth):
            b[r][c] = p
            try:
                if self._wins(b, r, c, p):
                    score = MATE - ply
                else:
                    score = -self._search(b, 3 - p, depth - 1, -beta, -alpha,
                                          empty_count - 1, ply + 1, extensions)
            finally:
                b[r][c] = 0
            value = max(value, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return value

    def get_move(self, board, last_opponent_move, time_limit):
        start = time.perf_counter()
        budget = max(0.0, float(time_limit))
        self.deadline = start + max(0.0, budget - max(0.005, min(0.15, budget * 0.1)))
        b = [row[:] for row in board]
        n, p = self.board_size, self.player_id
        empties = [(r, c) for r in range(n) for c in range(n) if b[r][c] == 0]
        if not empties:
            raise ValueError("No legal move on a full board")
        best = min(empties, key=lambda m: abs(m[0] - n // 2) + abs(m[1] - n // 2))
        self.last_depth, self.nodes = 0, 0
        self._tactic_cache.clear()
        if len(empties) == n * n or len(empties) == 1:
            return best
        try:
            # Check every empty cell before selective search; never omit an
            # immediate win or necessary block because of the candidate cap.
            blocks = []
            for r, c in empties:
                self._check_time()
                if self._wins(b, r, c, p):
                    return (r, c)
                if self._wins(b, r, c, 3 - p):
                    blocks.append((r, c))
            if len(blocks) == 1:
                return blocks[0]
            moves = self._candidates(b, p, 1)
            best = moves[0]
            _, own_pairs, _ = self._tactics(b, p)
            forks = [m for m, points in own_pairs.items() if len(points) >= 2]
            if not blocks and forks:
                return min(forks)
            _, enemy_pairs, _ = self._tactics(b, 3 - p)
            if not blocks and any(len(points) >= 2 for points in enemy_pairs.values()):
                # Keep quiet moves only if they remove ALL opponent winning
                # forks. A forcing four may legitimately take the initiative.
                safe = []
                for r, c in moves:
                    self._check_time()
                    b[r][c] = p
                    try:
                        forcing = self._tactics(b, p)[0]
                        remaining = self._tactics(b, 3 - p)[1]
                        if forcing or not any(len(points) >= 2 for points in remaining.values()):
                            safe.append((r, c))
                            # Save immediately: a timeout during this filtering
                            # pass must still return a verified defensive move.
                            if len(safe) == 1:
                                best = safe[0]
                    finally:
                        b[r][c] = 0
                if safe:
                    moves, best = safe, safe[0]
            for depth in range(1, min(self.max_depth, len(empties)) + 1):
                alpha, chosen = -float("inf"), best
                ordered = [best] + [m for m in moves if m != best]
                for r, c in ordered:
                    self._check_time()
                    b[r][c] = p
                    try:
                        score = -self._search(b, 3 - p, depth - 1,
                                              -float("inf"), -alpha, len(empties) - 1, 1)
                    finally:
                        b[r][c] = 0
                    if score > alpha:
                        alpha, chosen = score, (r, c)
                best, self.last_depth = chosen, depth
                if abs(alpha) > MATE / 2:
                    break
        except _Deadline:
            pass
        return best
