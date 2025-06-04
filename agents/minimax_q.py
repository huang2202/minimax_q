"""
Minimax-Q Learning implementation for a two-player zero-sum board game
(8×8 MiniGo, one-stone capture).

简化说明
--------
* 我们把每一方都视为“自私理性”智能体，各自维护一张独立的 Q-table。
* 环境返回的 reward 已经以“当前落子方 perspective”给出：
      - +1  : 我方赢
      -  0  : 平局 / 未终局
      - -10 : 非法落子（直接判负）
      - 其他负数或正数可自行扩展
* 对于零和博弈，另一方的 reward 就是 `-reward`，
  故在训练循环里逆序回放时直接取相反数即可。
* Minimax-V(s) = min_a' Q(s, a')  (对手最优假设)。
  这里采用 Littman1994 的“学习率分离”形式：
      Q(s,a) ← (1-α)Q(s,a) + α( r + γ·V(s') )
  其中 V(s') = {   max_a' Q(s',a')  若下一回合还是我下
                | min_a' Q(s',a')  若下一回合轮到对手下 }
  由于我们为“对手”单独训练另一张表，所以
  这里可用普通 Q-learning 公式简化 —— 收敛到
  Nash equilibrium 的性质由双智能体自对弈保证。
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Tuple, Hashable

import numpy as np

StateKey = Hashable            # bytes produced by ndarray.tobytes()
Action   = Tuple[int, int]     # (x, y) coordinate on the board


class MinimaxQLearning:
    """
    一个简洁、可扩展的 Minimax-Q 学习器。
    """

    def __init__(
        self,
        alpha: float = 0.6,
        gamma: float = 0.95,
        epsilon: float = 0.3,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int | None = None,
    ):
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        self._rng = random.Random(seed)
        # Q-table：dict[state_key][action] = value
        self.q_table: Dict[StateKey, Dict[Action, float]] = defaultdict(dict)

    # ------------------------------------------------------------------ #
    #                   公共接口                                          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _state_key(state: np.ndarray) -> StateKey:
        """
        将 8×8 numpy int board 压缩成 hash key。
        可按需添加其它特征（如当前执棋方）来避免冲突。
        """
        return state.tobytes()

    def choose_action(
        self,
        state: np.ndarray,
        available_actions: List[Action],
        explore: bool = True,
    ) -> Action:
        """
        ε-greedy 选动作。`explore=False` 用于评估阶段。
        """
        if explore and self._rng.random() < self.epsilon:
            return self._rng.choice(available_actions)

        state_key = self._state_key(state)
        q_values  = self.q_table.get(state_key, {})

        # 未见过的动作 Q=0；多最大值随机
        best_q = float("-inf")
        best_actions = []
        for a in available_actions:
            q = q_values.get(a, 0.0)
            if q > best_q + 1e-9:
                best_q, best_actions = q, [a]
            elif abs(q - best_q) <= 1e-9:
                best_actions.append(a)
        return self._rng.choice(best_actions)

    def update_q_value(
        self,
        state: np.ndarray,
        action: Action,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        current_player: int,
    ) -> None:
        """
        Q(s,a) ← (1-α)Q(s,a) + α( r + γ·max_a' Q(s',a') )  (若 not done)
        终局则无 bootstrap。
        current_player: 1 (黑) or -1 (白)，主要保留扩展空间——
        如果想把“轮到谁”编码进 state_key，可在 _state_key 里组合。
        """
        s_key  = self._state_key(state)
        ns_key = self._state_key(next_state)

        # 初始化不存在的动作 Q
        if action not in self.q_table[s_key]:
            self.q_table[s_key][action] = 0.0

        q_sa = self.q_table[s_key][action]

        if done:
            target = reward
        else:
            # Bootstrap：max_a' Q(s',a')
            next_qs = self.q_table.get(ns_key, {})
            max_next_q = max(next_qs.values(), default=0.0)
            target = reward + self.gamma * max_next_q

        # 更新
        self.q_table[s_key][action] = (1 - self.alpha) * q_sa + self.alpha * target

    # --------------------------- 其他工具 ------------------------------- #
    def decay_epsilon(self) -> None:
        """在训练过程中周期性调用，逐步减少探索。"""
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    # 供外部保存 / 加载 --------------------------------------------------- #
    def get_serializable_data(self) -> dict:
        """返回可 JSON / pickle 序列化的内部状态。"""
        return {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "q_table": self.q_table,
        }

    def load_from_data(self, data: dict) -> None:
        """从 dict 恢复权重；常用于 checkpoint 读取。"""
        self.alpha         = data["alpha"]
        self.gamma         = data["gamma"]
        self.epsilon       = data["epsilon"]
        self.epsilon_min   = data["epsilon_min"]
        self.epsilon_decay = data["epsilon_decay"]
        self.q_table       = defaultdict(dict, {k: dict(v) for k, v in data["q_table"].items()})
    
    def get_available_actions(self, state):
        return [(i, j) for i in range(state.shape[0]) for j in range(state.shape[1]) if state[i, j] == 0]
