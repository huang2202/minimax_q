#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiniGo-8×8 课程作业命令行入口
────────────────────────────
• 子命令
   train  : 自对弈训练
   eval   : 评估模型（统计胜率 + 热力图）
   play   : 人机对弈（可选）

• 依赖 utils.io + utils.plotting
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, List

import numpy as np
from tqdm import trange

from agents.minimax_q import MinimaxQLearning
from game.env import MiniGoEnv
from trainers.train import train as train_loop
from utils.io import load_agents, save_agents
from utils.plotting import plot_board_heatmap

# ------------------------------------------------------------
# 文本显示辅助
# ------------------------------------------------------------
_SYMBOL = {0: ".", 1: "X", -1: "O"}


def _print_board_ascii(board: np.ndarray) -> None:
    for row in board:
        print(" ".join(_SYMBOL[int(v)] for v in row))
    print()


# ------------------------------------------------------------
# 自对弈帮助函数
# ------------------------------------------------------------
def _self_play_one(
    agent_b: MinimaxQLearning,
    agent_w: MinimaxQLearning,
    env: MiniGoEnv,
) -> Tuple[int, List[Tuple[int, int]]]:
    """
    自对弈一局
    Returns
    -------
    reward : int
        黑胜=+1, 白胜=-1, 平=0
    moves  : list[(x, y)]
        全部落子序列，用于热力图
    """
    state = env.reset()
    done = False
    reward = 0
    moves: list[Tuple[int, int]] = []

    while not done:
        cp = env.current_player
        agent = agent_b if cp == 1 else agent_w
        action = agent.choose_action(state, env.board.legal_moves(), explore=False)
        moves.append(action)
        state, reward, done, _ = env.step(action)

    return (reward if cp == 1 else -reward), moves


# ------------------------------------------------------------
# 三大子命令
# ------------------------------------------------------------
def _cmd_train(args: argparse.Namespace) -> None:
    """
    调 trainers/train.py::train()
    自动 checkpoint 到 ckpt/
    """
    ckpt_dir = Path("ckpt")
    ckpt_dir.mkdir(exist_ok=True)

    # ───── 修正这里 ─────
    def _save(step: int, agents) -> None:
        agent_b, agent_w = agents          # ← 解包
        save_agents(agent_b, agent_w, ckpt_dir / f"ep{step}.pkl")
    # ───────────────────

    print(f"[Train] episodes={args.episodes}")
    train_loop(
        episodes=args.episodes,
        log_every=200,
        checkpoint_every=2_000,
        save_fn=_save,
    )
    print("✔ Training finished.")


def _cmd_eval(args: argparse.Namespace) -> None:
    print(f"[Eval] model={args.model}, games={args.games}")

    agent_b, agent_w = load_agents(args.model, MinimaxQLearning)
    env   = MiniGoEnv()
    stats = {"black": 0, "white": 0, "draw": 0}
    visits = np.zeros((8, 8), dtype=int)

    for _ in trange(args.games, desc="Evaluating"):
        reward, moves = _self_play_one(agent_b, agent_w, env)
        if reward == 1:
            stats["black"] += 1
        elif reward == -1:
            stats["white"] += 1
        else:
            stats["draw"] += 1
        for x, y in moves:
            visits[x, y] += 1

    total = args.games
    print("\n=== Self-play Result ===")
    for k, v in stats.items():
        print(f"{k:>5}: {v:4d}  ({v/total:.2%})")

    # ---------- 生成落子热力图 ----------
    plot_board_heatmap(
        visits,
        title=fr"\textbf{{Move frequency ({total} games)}}",
        annotate=True,
        save_path="fig/eval_heatmap",
        show=False,
    )
    print("✔ Heatmap saved → fig/eval_heatmap.png")


def _cmd_play(args: argparse.Namespace) -> None:
    print(f"[Play] you are {args.human.upper()}")

    agent_b, agent_w = load_agents(args.model, MinimaxQLearning)
    env = MiniGoEnv()
    human_color = 1 if args.human.lower() == "black" else -1

    state = env.reset()
    done = False
    reward = 0
    while not done:
        if env.current_player == human_color:
            _print_board_ascii(state)
            try:
                x, y = map(int, input("Your move (row col): ").split())
            except Exception:
                print("✗ 输入格式应为: 行 列（空格分隔），例如 3 4")
                continue
            action = (x, y)
        else:
            agent = agent_b if env.current_player == 1 else agent_w
            action = agent.choose_action(state, env.board.legal_moves(), explore=False)
            print(f"AI ({'Black' if env.current_player==1 else 'White'}) → {action}")

        state, reward, done, _ = env.step(action)

    _print_board_ascii(state)
    if reward == 0:
        print("＝ 平局 ＝")
    else:
        winner = "Black" if reward == 1 else "White"
        if (winner == "Black" and human_color == 1) or (winner == "White" and human_color == -1):
            print("🎉 你赢了！")
        else:
            print("😢 AI 获胜")

# ------- 顶部 import -------
from agents.mcts_light import MCTSLight          # ★ 新引入

# ------------------------------------------------------------
def _cmd_eval_vs(args: argparse.Namespace) -> None:
    """
    训练好的 Black Minimax-Q VS White MCTS-Light
    统计胜率
    """
    print(f"[Eval-VS] model={args.model}, games={args.games}, baseline=MCTS-Light")

    agent_black, _ = load_agents(args.model, MinimaxQLearning)
    agent_white    = MCTSLight()                 # 白方为静态 MCTS
    env   = MiniGoEnv()
    wins  = {"black": 0, "white": 0, "draw": 0}

    for _ in trange(args.games, desc="VS baseline"):
        state = env.reset()
        done  = False
        reward = 0
        while not done:
            cp = env.current_player
            legal = env.board.legal_moves()
            if cp == 1:   # Black = 学习好的
                action = agent_black.choose_action(state, legal, explore=False)
            else:        # White = Baseline
                action = agent_white.choose_action(state, legal, env)
            state, reward, done, _ = env.step(action)

        reward = reward if cp == 1 else -reward
        if reward == 1:
            wins["black"] += 1
        elif reward == -1:
            wins["white"] += 1
        else:
            wins["draw"]  += 1

    total = args.games
    print("\n=== VS MCTS-Light Result ===")
    for k, v in wins.items():
        print(f"{k:>5}: {v:4d}  ({v/total:.2%})")



# ------------------------------------------------------------
# CLI 构建
# ------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MiniGo-8×8 Coursework CLI")
    sub = p.add_subparsers(dest="subcmd", required=True)

    # ---- train ----
    p_train = sub.add_parser("train", help="Self-play training")
    p_train.add_argument("--episodes", type=int, default=10_000)
    p_train.set_defaults(func=_cmd_train)

    # ---- eval ----
    p_eval = sub.add_parser("eval", help="Evaluate a saved model")
    p_eval.add_argument("--model", type=Path, required=True)
    p_eval.add_argument("--games", type=int, default=500)
    p_eval.set_defaults(func=_cmd_eval)

    # ---- play ----
    p_play = sub.add_parser("play", help="Play against the trained AI")
    p_play.add_argument("--model", type=Path, required=True)
    p_play.add_argument(
        "--human", choices=["black", "white"], default="black", help="Which color you play"
    )
    p_play.set_defaults(func=_cmd_play)

    # in _build_parser()
    p_eval_vs = sub.add_parser("eval_vs", help="Evaluate against static MCTS-Light baseline")
    p_eval_vs.add_argument("--model", type=Path, required=True)
    p_eval_vs.add_argument("--games", type=int, default=500)
    p_eval_vs.set_defaults(func=_cmd_eval_vs)

    return p


def main() -> None:
    args = _build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


# python main.py train --episodes 20000
# python main.py eval  --model ckpt_ep19500.pkl
# python main.py play  --model ckpt_ep14000.pkl --human white
