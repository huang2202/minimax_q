"""
可视化工具：

1. plot_reward_curve(rewards, ...)       —— 绘制训练累计/平滑奖励曲线
2. plot_board_heatmap(board, ...)        —— 绘制 8×8 围棋棋盘着子热图
3. quick_compare_curves(curves_dict, ...)—— 多条曲线对比（可用于不同超参）

字体 & 颜色风格参照 sr_pic.py，保证论文/汇报统一视觉。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

# ------------------------------------------------------------------ #
#           全局美化：LaTeX + Times New Roman + 统一色系              #
# ------------------------------------------------------------------ #
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stixsans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# 取 sr_pic.py 中常用的配色（伯爵橙 / 奶茶粉 / 枫舞灰）
PALETTE = {
    "reward_raw": "#e4cece",   # 奶茶粉
    "reward_avg": "#e38c7a",   # 伯爵橙
    "reward_other": "#dccfcb", # 枫舞灰
    "board_bg": "#f6f1e0",     # 香草黄
    "board_black": "#000000",
    "board_white": "#ffffff",
}

# ------------------------------------------------------------------ #
#                           核心函数                                 #
# ------------------------------------------------------------------ #
def moving_average(x: Sequence[float], window: int) -> np.ndarray:
    """简单滑动平均，首尾填充保持长度不变。"""
    x = np.asarray(x, dtype=float)
    if window <= 1:
        return x
    cumsum = np.cumsum(np.insert(x, 0, 0))
    ma = (cumsum[window:] - cumsum[:-window]) / window
    pad_left = np.full(window - 1, ma[0])
    return np.concatenate([pad_left, ma])


def plot_reward_curve(
    rewards: Sequence[float],
    window: int = 200,
    figsize: tuple[int, int] = (8, 4),
    title: str | None = r"\textbf{Training Reward}",
    save_path: str | Path | None = None,
    show: bool = True,
):
    """
    绘制奖励曲线（原始 + 平滑）。
    Parameters
    ----------
    rewards    : 每局/每步 reward 序列
    window     : 平滑窗口
    figsize    : 图像尺寸
    title      : 图标题
    save_path  : 保存 *.png / *.pdf 路径（可省略）
    show       : 是否调用 plt.show()
    """
    rewards = np.asarray(rewards, dtype=float)
    avg = moving_average(rewards, window)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(rewards))

    ax.plot(
        x,
        rewards,
        label=rf"\textbf{{Raw}}",
        color=PALETTE["reward_raw"],
        alpha=0.4,
        linewidth=1,
    )
    ax.plot(
        x,
        avg,
        label=rf"\textbf{{{window}-step MA}}",
        color=PALETTE["reward_avg"],
        linewidth=2,
    )
    ax.set_xlabel(r"\textbf{Episode}", fontsize=11)
    ax.set_ylabel(r"\textbf{Reward}", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.3)
    if title:
        ax.set_title(title, fontsize=13, pad=10)
    ax.legend(frameon=False, fontsize=10)

    plt.tight_layout()
    if save_path:
        Path(save_path).with_suffix(".png").parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(Path(save_path).with_suffix(".png")), dpi=300, bbox_inches="tight")
        fig.savefig(str(Path(save_path).with_suffix(".pdf")), bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig, ax


def plot_board_heatmap(
    board: np.ndarray,
    title: str | None = r"\textbf{Board State}",
    cmap: str = "coolwarm",
    annotate: bool = False,
    save_path: str | Path | None = None,
    show: bool = True,
):
    """
    根据 8×8 棋盘矩阵绘制热图：
        1  → 黑子
       -1  → 白子
        0  → 空
    其他数值（如落子频次）也可直接可视化。
    """
    fig, ax = plt.subplots(figsize=(4, 4))

    # 画背景格网
    ax.imshow(
        np.full_like(board, np.nan, dtype=float),
        cmap="gray",
        vmin=0,
        vmax=1,
        alpha=0,
    )
    for i in range(board.shape[0] + 1):
        ax.axhline(i - 0.5, color="#999999", linewidth=0.5, alpha=0.6)
        ax.axvline(i - 0.5, color="#999999", linewidth=0.5, alpha=0.6)

    # 着子：黑/白/空
    for (x, y), val in np.ndenumerate(board):
        if val == 1:
            circle = plt.Circle((y, x), 0.38, color=PALETTE["board_black"])
            ax.add_patch(circle)
        elif val == -1:
            circle = plt.Circle((y, x), 0.38, color=PALETTE["board_white"], ec="black", linewidth=0.8)
            ax.add_patch(circle)
        elif not math.isclose(val, 0):  # 显示数值型热度
            ax.text(y, x, f"{val:.1f}", ha="center", va="center", fontsize=7)

    ax.set_aspect("equal")
    ax.set_xlim(-0.5, board.shape[1] - 0.5)
    ax.set_ylim(board.shape[0] - 0.5, -0.5)  # 原点左上
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=12, pad=6)

    plt.tight_layout()
    if save_path:
        Path(save_path).with_suffix(".png").parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(Path(save_path).with_suffix(".png")), dpi=300, bbox_inches="tight")
        fig.savefig(str(Path(save_path).with_suffix(".pdf")), bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig, ax


def quick_compare_curves(
    curves: Dict[str, Sequence[float]],
    window: int = 200,
    figsize: tuple[int, int] = (8, 4),
    save_path: str | Path | None = None,
    show: bool = True,
):
    """
    将多条训练曲线放在一张图中快速对比。
    Parameters
    ----------
    curves   : {"label": rewards_list}
    window   : 每条曲线单独平滑
    """
    fig, ax = plt.subplots(figsize=figsize)

    for i, (label, rewards) in enumerate(curves.items()):
        color = list(PALETTE.values())[(i + 1) % len(PALETTE)]
        rewards = np.asarray(rewards, dtype=float)
        ax.plot(
            moving_average(rewards, window),
            label=rf"\textbf{{{label}}}",
            linewidth=2,
            color=color,
        )

    ax.set_xlabel(r"\textbf{Episode}", fontsize=11)
    ax.set_ylabel(r"\textbf{Smoothed Reward}", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()

    if save_path:
        Path(save_path).with_suffix(".png").parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(Path(save_path).with_suffix(".png")), dpi=300, bbox_inches="tight")
        fig.savefig(str(Path(save_path).with_suffix(".pdf")), bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig, ax
