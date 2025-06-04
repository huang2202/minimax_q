#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自对弈训练入口（MiniGo-8×8）
────────────────────────────
• 负责：
  1. 重复 self-play 更新 Q-table
  2. 每 log_every 局打印一次窗口胜率
  3. 周期性自动 checkpoint（由外部传入 save_fn）
  4. 训练结束后：
       ─ 保存逐局 reward 到 .npy
       ─ 绘制学习曲线 fig/train_curve.{png,pdf}

• 依赖 utils.io.save_numpy_log / utils.plotting.plot_reward_curve
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from tqdm import trange

from agents.minimax_q import MinimaxQLearning
from game.env import MiniGoEnv
from utils.io import save_numpy_log
from utils.plotting import plot_reward_curve

# ------------------------------------------------------------
# 基础工具
# ------------------------------------------------------------

def _self_play(
    agent_b: MinimaxQLearning,
    agent_w: MinimaxQLearning,
    env: MiniGoEnv,
) -> int:
    """
    进行一局自对弈，返回 reward：
      黑胜=+1, 白胜=-1, 平=0
    """
    state = env.reset()
    done = False
    reward = 0

    while not done:
        cp = env.current_player
        agent = agent_b if cp == 1 else agent_w
        action = agent.choose_action(state, env.board.legal_moves(), explore=True)
        state, reward, done, _ = env.step(action)

    return reward if cp == 1 else -reward


# ------------------------------------------------------------
# 公开的 train() 接口
# ------------------------------------------------------------

def train(
    episodes: int = 10_000,
    log_every: int = 200,
    checkpoint_every: int = 5_000,
    save_fn: Callable[[int, Sequence[MinimaxQLearning]], None] | None = None,
) -> None:
    """
    Parameters
    ----------
    episodes : int
        总对局数
    log_every : int
        窗口胜率 & 进度条打印间隔
    checkpoint_every : int
        自动调用 save_fn 的间隔
    save_fn : Callable
        save_fn(step, [agent_b, agent_w])
    """
    env = MiniGoEnv()
    agent_b = MinimaxQLearning()
    agent_w = MinimaxQLearning()

    rewards: list[int] = []
    win_window: list[int] = []

    t0 = time.time()
    for ep in trange(1, episodes + 1, desc="Training"):
        r = _self_play(agent_b, agent_w, env)
        rewards.append(r)

        # --- 滚动窗口平均 ---
        win_window.append(r)
        if len(win_window) > log_every:
            win_window.pop(0)

        if ep % log_every == 0:
            print(f"[{ep:6d}/{episodes}] "
                  f"black_win_rate={np.mean(win_window):.3f}  "
                  f"elapsed={time.time()-t0:.1f}s")

        # --- 自动 checkpoint ---
        if save_fn is not None and ep % checkpoint_every == 0:
            save_fn(ep, (agent_b, agent_w))

    # ---------- 训练结束：写日志 & 画学习曲线 ----------
    Path("log").mkdir(exist_ok=True)
    save_numpy_log(np.array(rewards), "log/reward_ep.npy")

    plot_reward_curve(
        rewards,
        window=log_every,
        save_path="fig/train_curve",
        show=False,
    )
    print("✔ Train curve saved → fig/train_curve.png")
