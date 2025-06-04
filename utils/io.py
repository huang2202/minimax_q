"""
模型与实验日志的序列化 / 反序列化工具。

* save_agents(agent_b, agent_w, path)     —— 保存双智能体
* load_agents(path, agent_cls)            —— 读取并返回 (agent_b, agent_w)
* save_numpy_log(arr, path)               —— 记录数值型数组（训练奖励等）
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Tuple, Type

import numpy as np


# ------------------------------------------------------------------ #
#                        Agent I/O                                   #
# ------------------------------------------------------------------ #
def save_agents(agent_black: Any, agent_white: Any, path: str | Path) -> Path:
    """
    将双方智能体序列化到磁盘。自动创建目录，返回实际保存路径。
    """
    path = Path(path).with_suffix(".pkl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(
            {
                "black": agent_black.get_serializable_data(),
                "white": agent_white.get_serializable_data(),
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return path


def load_agents(path: str | Path, agent_cls: Type) -> Tuple[Any, Any]:
    """
    读取智能体，自动实例化 `agent_cls()` 并 load_from_data。
    返回 (agent_black, agent_white)。
    """
    path = Path(path)
    with path.open("rb") as f:
        data = pickle.load(f)

    agent_b, agent_w = agent_cls(), agent_cls()
    agent_b.load_from_data(data["black"])
    agent_w.load_from_data(data["white"])
    return agent_b, agent_w


# ------------------------------------------------------------------ #
#                    训练日志 / 数值结果 I/O                         #
# ------------------------------------------------------------------ #
def save_numpy_log(arr: np.ndarray, path: str | Path) -> Path:
    """
    将 NumPy 数组保存为 .npy，便于后期可重复绘图或分析。
    """
    path = Path(path).with_suffix(".npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    return path


def load_numpy_log(path: str | Path) -> np.ndarray:
    """
    读取由 save_numpy_log 保存的数组。
    """
    return np.load(Path(path), allow_pickle=False)
