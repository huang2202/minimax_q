import numpy as np
from collections import deque

class MiniGoBoard:
    """8×8 一子提子版 Atari-Go：先提到对方 1 颗子即胜，平局=落满棋盘"""
    SIZE = 8
    EMPTY, BLACK, WHITE = 0, 1, -1

    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((self.SIZE, self.SIZE), dtype=int)
        self.history = set()           # 打劫判定：哈希后的棋盘记录
        self.captured = {self.BLACK:0, self.WHITE:0}
        return self.board.copy()

    # ---------- 内部工具 ----------
    def _neighbors(self, x, y):
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx, ny = x+dx, y+dy
            if 0 <= nx < self.SIZE and 0 <= ny < self.SIZE:
                yield nx, ny

    def _group_and_liberties(self, x, y):
        color = self.board[x,y]
        group, libs = set(), set()
        queue = deque([(x,y)])
        while queue:
            i,j = queue.popleft()
            group.add((i,j))
            for nx, ny in self._neighbors(i,j):
                if self.board[nx,ny] == self.EMPTY:
                    libs.add((nx,ny))
                elif self.board[nx,ny] == color and (nx,ny) not in group:
                    queue.append((nx,ny))
        return group, libs

    # ---------- 外部接口 ----------
    def play(self, x, y, color):
        """尝试落子；返回(合法? bool, 奖励, 终局?)"""
        if self.board[x,y] != self.EMPTY:
            return False, -10, True    # 非法落子直接判负

        snapshot = self.board.copy()
        self.board[x,y] = color

        # 提子逻辑：检查邻接对方棋块
        opponent = -color
        captured_now = 0
        for nx, ny in self._neighbors(x,y):
            if self.board[nx,ny] == opponent:
                grp, libs = self._group_and_liberties(nx,ny)
                if not libs:           # 无气被提
                    for gx,gy in grp:
                        self.board[gx,gy] = self.EMPTY
                    captured_now += len(grp)

        # 自杀 & 打劫：若自己无气或局面重复，恢复快照
        _, self_libs = self._group_and_liberties(x,y)
        board_hash = self._hash()
        if not self_libs or board_hash in self.history:
            self.board = snapshot
            return False, -10, True

        self.history.add(board_hash)
        self.captured[color] += captured_now

        # 终局：先提 1 子即胜
        if captured_now > 0:
            return True, 1, True
        if np.all(self.board != self.EMPTY):
            return True, 0, True        # 平局
    
        if not captured_now and not self.EMPTY in self.board:
            return True, 0, True        # 平局
        
        return True, 0, False

    def legal_moves(self):
        return [(i,j) for i in range(self.SIZE) for j in range(self.SIZE)
                if self.board[i,j] == self.EMPTY]

    def _hash(self):
        return hash(self.board.tobytes())
    
