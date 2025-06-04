from __future__ import annotations
from collections import defaultdict
import math
import random
import copy
from typing import Any, Tuple, List

import numpy as np
from game.env import MiniGoEnv


class MCTSLight:
    """
    极简 Monte-Carlo Tree Search baseline
    - 模拟 N_ROLLOUTS 次随机对局
    - UCT 选子
    - 不学习，评估时作为静态对手
    """

    N_ROLLOUTS = 100
    C_PUCT = 1.4

    def __init__(self):
        self.Q = defaultdict(float)   # 累积 reward
        self.N = defaultdict(int)     # 访问次数

    # --------------------------------------------------------

    def _uct(self, s_hash: int, action_hashes: List[int]) -> int:
        total = sum(self.N[(s_hash, a)] for a in action_hashes)
        best, best_a = -1e9, action_hashes[0]
        for a in action_hashes:
            q = self.Q[(s_hash, a)]
            n = self.N[(s_hash, a)]
            ucb = q / (n + 1e-6) + self.C_PUCT * math.sqrt(
                math.log(total + 1) / (n + 1e-6)
            )
            if ucb > best:
                best, best_a = ucb, a
        return best_a

    # --------------------------------------------------------

    def _random_rollout(self, env: MiniGoEnv) -> int:
        """一路随机走到终局，返回黑赢+1/白赢-1/平0"""
        done = False
        reward = 0
        while not done:
            legal = env.board.legal_moves()
            action = random.choice(legal)
            _, reward, done, _ = env.step(action)
        cp = env.current_player
        return reward if cp == 1 else -reward

    # --------------------------------------------------------

    def _clone_env(self, env: MiniGoEnv) -> MiniGoEnv:
        """优先用 env.copy()；若无则 deepcopy"""
        if hasattr(env, "copy"):
            return env.copy()            # type: ignore[attr-defined]
        return copy.deepcopy(env)

    # --------------------------------------------------------

    def choose_action(
        self,
        state: np.ndarray,
        legal_moves: List[Tuple[int, int]],
        env: MiniGoEnv,
    ) -> Tuple[int, int]:
        """给定当前 env（引用）选择动作"""
        if len(legal_moves) == 1:
            return legal_moves[0]

        s_hash = hash(state.tobytes())

        # 每个合法动作做 N_ROLLOUTS 次模拟
        for move in legal_moves:
            for _ in range(self.N_ROLLOUTS):
                env_copy = self._clone_env(env)
                env_copy.step(move)
                r = self._random_rollout(env_copy)

                key = (s_hash, hash(move))
                self.N[key] += 1
                self.Q[key] += r

        best_hash = self._uct(s_hash, [hash(a) for a in legal_moves])
        for a in legal_moves:
            if hash(a) == best_hash:
                return a
        return legal_moves[0]  # 理论不会走到
