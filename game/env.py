from game.board import MiniGoBoard

class MiniGoEnv:
    """Gym-style 封装，公平双人对弈。"""
    def __init__(self):
        self.board = MiniGoBoard()
        self.current_player = MiniGoBoard.BLACK  # 黑先

    def reset(self):
        self.current_player = MiniGoBoard.BLACK
        return self.board.reset()

    def step(self, action):
        x, y = action
        ok, reward, done = self.board.play(x, y, self.current_player)
        if not ok:
            return self.board.board.copy(), reward, done, {}
        if not done:
            self.current_player *= -1
        return self.board.board.copy(), reward, done, {}
