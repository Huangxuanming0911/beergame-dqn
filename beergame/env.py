from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BeerGameConfig:
    """环境参数集中放在一个对象里，便于复现实验。"""

    num_firms: int = 3
    prices: tuple[float, ...] = (10.0, 9.0, 8.0)
    holding_cost: float = 0.5
    lost_sales_cost: float = 2.0
    initial_inventory: int = 100
    poisson_lambda: float = 10.0
    max_steps: int = 100
    max_order: int = 20
    seed: int | None = 42


class BeerGameEnv:
    """串行啤酒游戏供应链环境。

    企业0面对外部顾客需求，企业i>0面对下游企业i-1的订单。
    t时刻发出的订单会在t+1时刻进入库存。
    观测保持课程给定格式：[上一期订货量, 上一期满足需求量, 当前库存]。
    """

    def __init__(self, config: BeerGameConfig):
        self.config = config
        if len(config.prices) != config.num_firms:
            raise ValueError("prices length must equal num_firms")
        self.rng = np.random.default_rng(config.seed)
        self.reset()

    @property
    def num_firms(self) -> int:
        return self.config.num_firms

    @property
    def max_steps(self) -> int:
        return self.config.max_steps

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        n = self.config.num_firms
        self.inventory = np.full(n, self.config.initial_inventory, dtype=np.float32)
        # pipeline[i] 表示企业i将在下一步收到的在途货物。
        self.pipeline = np.zeros(n, dtype=np.float32)
        self.last_orders = np.zeros(n, dtype=np.float32)
        self.demand = np.zeros(n, dtype=np.float32)
        self.satisfied_demand = np.zeros(n, dtype=np.float32)
        self.current_step = 0
        self.done = False
        return self._observation()

    def _observation(self) -> np.ndarray:
        # 每个企业只观测自己的局部状态。
        return np.stack(
            [self.last_orders, self.satisfied_demand, self.inventory],
            axis=1,
        ).astype(np.float32)

    def _clip_actions(self, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32).reshape(self.config.num_firms)
        return np.clip(np.rint(actions), 0, self.config.max_order).astype(np.float32)

    def step(self, actions: np.ndarray):
        if self.done:
            raise RuntimeError("step() called after episode is done")

        actions = self._clip_actions(actions)
        n = self.config.num_firms

        # t+1到货：当前时刻只能收到上一期已经进入pipeline的货物。
        inbound = self.pipeline.copy()
        self.inventory += inbound

        # 企业0面对外部顾客需求；更上游企业面对下游企业的订单。
        demand = np.zeros(n, dtype=np.float32)
        demand[0] = self.rng.poisson(self.config.poisson_lambda)
        if n > 1:
            demand[1:] = actions[:-1]

        # 实际销售受当前库存限制，未满足部分计入缺货损失。
        satisfied = np.minimum(demand, self.inventory)
        self.inventory -= satisfied

        # 本期产生的发货进入pipeline，下一期才会到达对应企业。
        next_pipeline = np.zeros(n, dtype=np.float32)
        if n > 1:
            next_pipeline[:-1] = satisfied[1:]
        next_pipeline[-1] = actions[-1]

        rewards = np.zeros(n, dtype=np.float32)
        prices = np.asarray(self.config.prices, dtype=np.float32)
        for i in range(n):
            # 利润 = 销售收入 - 采购成本 - 库存持有成本 - 缺货惩罚。
            revenue = prices[i] * satisfied[i]
            purchase_cost = prices[i + 1] * actions[i] if i + 1 < n else 0.0
            holding_cost = self.config.holding_cost * self.inventory[i]
            lost_sales = self.config.lost_sales_cost * max(demand[i] - satisfied[i], 0.0)
            rewards[i] = revenue - purchase_cost - holding_cost - lost_sales

        self.pipeline = next_pipeline
        self.last_orders = actions
        self.demand = demand
        self.satisfied_demand = satisfied
        self.current_step += 1
        self.done = self.current_step >= self.config.max_steps

        # info只用于记录和画图，不属于智能体实际可见的观测。
        info = {
            "actions": actions.copy(),
            "demand": demand.copy(),
            "satisfied_demand": satisfied.copy(),
            "inventory": self.inventory.copy(),
            "inbound": inbound.copy(),
            "pipeline": self.pipeline.copy(),
        }
        return self._observation(), rewards.reshape(n, 1), self.done, info
