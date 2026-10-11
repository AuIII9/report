"""Tk desktop UI. AI work stays off the Tk event thread."""
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from gomoku_game import Game, load_ai, timed_move


class App:
    def __init__(self, root):
        self.root = root
        root.title("广义五子棋 · Gomoku")
        root.geometry("1030x790")
        root.minsize(840, 640)
        root.configure(bg="#f3f5f7")
        self.results = queue.Queue()
        self.generation = 0
        self.game = None
        self.busy = False
        self.ais = {}
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#f3f5f7")
        style.configure("TLabel", background="#f3f5f7", font=("Microsoft YaHei UI", 10))
        style.configure("TButton", font=("Microsoft YaHei UI", 10), padding=7)
        shell = ttk.Frame(root, padding=18)
        shell.pack(fill="both", expand=True)
        left = ttk.Frame(shell)
        left.pack(side="left", fill="both", expand=True)
        ttk.Label(left, text="广义五子棋", font=("Microsoft YaHei UI", 22, "bold")).pack(anchor="w")
        ttk.Label(left, text="横、竖或斜向连成 K 子获胜 · 无禁手 · 长连有效").pack(anchor="w", pady=(4, 12))
        self.canvas = tk.Canvas(left, background="#dfbd87", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.draw())
        self.canvas.bind("<Button-1>", self.click)
        self.status = tk.StringVar(value="设置参数后，点击“开始 / 重新开始”。")
        ttk.Label(left, textvariable=self.status, wraplength=600).pack(anchor="w", pady=12)
        side = ttk.Frame(shell, padding=(20, 0, 0, 0))
        side.pack(side="right", fill="y")
        ttk.Label(side, text="对局设置", font=("Microsoft YaHei UI", 15, "bold")).pack(anchor="w", pady=(5, 15))
        self.mode = tk.StringVar(value="人机对战")
        self.size = tk.StringVar(value="15")
        self.win = tk.StringVar(value="5")
        self.limit = tk.StringVar(value="5")
        self.human = tk.StringVar(value="黑棋（先手）")
        self.black = tk.StringVar(value="builtin")
        self.white = tk.StringVar(value="builtin")
        for title, var, choices in (
            ("对战模式", self.mode, ("人人对战", "人机对战", "机机对战")),
            ("人类执棋（人机模式）", self.human, ("黑棋（先手）", "白棋（后手）")),
        ):
            ttk.Label(side, text=title).pack(anchor="w")
            ttk.Combobox(side, textvariable=var, values=choices, state="readonly", width=23).pack(fill="x", pady=(3, 10))
        for title, var in (("棋盘边长 N", self.size), ("获胜连珠数 K", self.win), ("AI 每步时限（秒）", self.limit)):
            ttk.Label(side, text=title).pack(anchor="w")
            ttk.Entry(side, textvariable=var, width=25).pack(fill="x", pady=(3, 10))
        for title, var in (("黑棋 AI", self.black), ("白棋 AI", self.white)):
            ttk.Label(side, text=title).pack(anchor="w")
            ttk.Entry(side, textvariable=var, width=25).pack(fill="x", pady=3)
            ttk.Button(side, text="选择 AI 文件…", command=lambda v=var: self.pick(v)).pack(fill="x", pady=(0, 8))
        ttk.Label(side, text="builtin：搜索 AI\nrandom：随机 AI\n也可选择符合协议的 .py 文件", foreground="#596575").pack(anchor="w", pady=5)
        ttk.Button(side, text="开始 / 重新开始", command=self.start).pack(fill="x", pady=(15, 5))
        ttk.Label(side, text="设置在下一局生效。\n点击交叉点落子，坐标从 0 开始。", foreground="#596575").pack(anchor="w", pady=10)
        self.root.after(40, self.poll)

    def pick(self, variable):
        path = filedialog.askopenfilename(title="选择 GomokuAI 文件", filetypes=[("Python", "*.py")])
        if path:
            variable.set(path)

    def start(self):
        try:
            n, k, limit = int(self.size.get()), int(self.win.get()), float(self.limit.get())
            if not 5 <= n <= 15:
                raise ValueError("图形界面棋盘大小请输入 5–15；更大棋盘可使用 CLI。")
            if not 0 < limit <= 60:
                raise ValueError("图形界面每步时限请输入大于 0、不超过 60 的秒数。")
            game = Game(n, k)
            ais = {}
            human = 1 if self.human.get().startswith("黑") else 2
            for p, var in ((1, self.black), (2, self.white)):
                if self.mode.get() == "机机对战" or (self.mode.get() == "人机对战" and p != human):
                    ais[p] = load_ai(var.get().strip())(p, n, k)
        except Exception as exc:
            messagebox.showerror("无法开始对局", str(exc))
            return
        self.generation += 1
        self.game, self.ais, self.seconds = game, ais, limit
        self.busy = False
        self.draw()
        self.advance()

    def geometry(self):
        n = self.game.size if self.game else 15
        space = max(1, (min(self.canvas.winfo_width(), self.canvas.winfo_height()) - 70) / (n - 1))
        x = (self.canvas.winfo_width() - space * (n - 1)) / 2
        y = (self.canvas.winfo_height() - space * (n - 1)) / 2
        return x, y, space

    def draw(self):
        cv = self.canvas
        cv.delete("all")
        n = self.game.size if self.game else 15
        x, y, s = self.geometry()
        end_x, end_y = x + (n - 1) * s, y + (n - 1) * s
        for i in range(n):
            cv.create_line(x, y + i * s, end_x, y + i * s, fill="#826444")
            cv.create_line(x + i * s, y, x + i * s, end_y, fill="#826444")
            cv.create_text(x - 20, y + i * s, text=str(i), fill="#5c4934")
            cv.create_text(x + i * s, y - 20, text=str(i), fill="#5c4934")
        if self.game:
            for r, row in enumerate(self.game.board):
                for c, p in enumerate(row):
                    if p:
                        cx, cy, radius = x + c * s, y + r * s, s * 0.41
                        cv.create_oval(cx - radius, cy - radius, cx + radius, cy + radius,
                                       fill="#20252c" if p == 1 else "#fffdf8", outline="#494c51", width=1.3)
            if self.game.history:
                r, c = self.game.history[-1]
                cx, cy = x + c * s, y + r * s
                cv.create_oval(cx - 3, cy - 3, cx + 3, cy + 3, fill="#e55745", outline="")

    def click(self, event):
        game = self.game
        if not game or game.winner is not None or self.busy or game.player in self.ais:
            return
        x, y, s = self.geometry()
        r, c = round((event.y - y) / s), round((event.x - x) / s)
        if abs(event.x - (x + c * s)) > s * 0.45 or abs(event.y - (y + r * s)) > s * 0.45:
            return
        try:
            game.play((r, c))
        except ValueError as exc:
            self.status.set(str(exc))
            return
        self.draw()
        self.advance()

    def advance(self, prefix=""):
        game = self.game
        if game.winner is not None:
            result = "和棋" if game.winner == 0 else ("黑棋" if game.winner == 1 else "白棋") + "获胜"
            self.status.set(f"{prefix}{result}！共 {len(game.history)} 步。")
            return
        p = game.player
        label = "黑棋" if p == 1 else "白棋"
        self.status.set(f"{prefix}第 {len(game.history) + 1} 步 · {label}" + (" AI 思考中…" if p in self.ais else "，请落子。"))
        if p in self.ais:
            self.busy = True
            token, ai, seconds = self.generation, self.ais[p], self.seconds
            board = [row[:] for row in game.board]
            last = game.history[-1] if game.history else None

            def work():
                result = timed_move(ai, board, last, seconds)
                self.results.put((token, result))

            threading.Thread(target=work, daemon=True).start()

    def poll(self):
        try:
            while True:
                token, (move, elapsed, error) = self.results.get_nowait()
                if token != self.generation:
                    continue
                self.busy = False
                try:
                    if error:
                        raise ValueError(error)
                    self.game.play(move)
                except ValueError as exc:
                    self.game.winner = 3 - self.game.player
                    self.advance(f"{exc}，判负。")
                    continue
                self.draw()
                self.advance(f"AI 用时 {elapsed:.2f}s。")
        except queue.Empty:
            pass
        self.root.after(40, self.poll)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
