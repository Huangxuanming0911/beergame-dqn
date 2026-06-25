from __future__ import annotations

import numpy as np


class RandomPolicy:
    """随机订货基线：在合法动作范围内均匀采样订单量。"""

    def __init__(self, max_order: int, seed: int | None = None):
        self.max_order = max_order
        self.rng = np.random.default_rng(seed)

    def act(self, state: np.ndarray, firm_id: int) -> int:
        return int(self.rng.integers(0, self.max_order + 1))


class BaseStockPolicy:
    """库存补足策略：订货到目标库存水位。"""

    def __init__(self, target_inventory: int, max_order: int):
        self.target_inventory = target_inventory
        self.max_order = max_order

    def act(self, state: np.ndarray, firm_id: int) -> int:
        # 该策略只使用被控制企业自己的当前库存。
        inventory = float(state[firm_id, 2])
        order = self.target_inventory - inventory
        return int(np.clip(round(order), 0, self.max_order))


def build_policy(name: str, max_order: int, seed: int | None = None, target_inventory: int = 100):
    if name == "random":
        return RandomPolicy(max_order=max_order, seed=seed)
    if name == "base_stock":
        return BaseStockPolicy(target_inventory=target_inventory, max_order=max_order)
    raise ValueError(f"unknown policy: {name}")
