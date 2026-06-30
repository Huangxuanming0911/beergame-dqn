from __future__ import annotations

import numpy as np

from beergame.env import BeerGameConfig


class VectorizedBeerGameEnv:
    """Vectorized beer-game environment running multiple independent episodes.

    The dynamics are implemented with batched NumPy operations so that a single
    ``step`` call advances all environments in parallel.  Environments that finish
    an episode are automatically reset, which makes the object convenient for
    collecting fixed-length rollout buffers for PPO.
    """

    def __init__(self, config: BeerGameConfig, num_envs: int, seed: int = 42):
        self.config = config
        self.num_envs = num_envs
        self.num_firms = config.num_firms
        self.max_steps = config.max_steps
        self.base_seed = seed
        self.episode_counts = np.zeros(num_envs, dtype=np.int64)
        self.rngs = [
            np.random.default_rng(seed + i) for i in range(num_envs)
        ]
        self.inventory: np.ndarray | None = None
        self.pipeline: np.ndarray | None = None
        self.last_orders: np.ndarray | None = None
        self.demand: np.ndarray | None = None
        self.satisfied_demand: np.ndarray | None = None
        self.current_step: np.ndarray | None = None
        self.done: np.ndarray | None = None
        self.reset()

    def reset(self) -> np.ndarray:
        n, m = self.num_envs, self.num_firms
        self.inventory = np.full((n, m), self.config.initial_inventory, dtype=np.float32)
        self.pipeline = np.zeros((n, m), dtype=np.float32)
        self.last_orders = np.zeros((n, m), dtype=np.float32)
        self.demand = np.zeros((n, m), dtype=np.float32)
        self.satisfied_demand = np.zeros((n, m), dtype=np.float32)
        self.current_step = np.zeros(n, dtype=np.int64)
        self.done = np.zeros(n, dtype=np.bool_)
        self.episode_counts = np.zeros(n, dtype=np.int64)
        return self._observation()

    def _observation(self) -> np.ndarray:
        return np.stack(
            [self.last_orders, self.satisfied_demand, self.inventory],
            axis=2,
        ).astype(np.float32)

    def _clip_actions(self, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32).reshape(
            self.num_envs, self.num_firms
        )
        return np.clip(np.rint(actions), 0, self.config.max_order).astype(np.float32)

    def _reset_envs(self, env_ids: np.ndarray) -> np.ndarray:
        """Reset the environments whose indices are provided."""
        if env_ids.size == 0:
            return self._observation()[env_ids]
        for idx in env_ids:
            self.rngs[idx] = np.random.default_rng(
                self.base_seed + idx + int(self.episode_counts[idx]) * self.num_envs
            )
            self.episode_counts[idx] += 1
        self.inventory[env_ids] = self.config.initial_inventory
        self.pipeline[env_ids] = 0.0
        self.last_orders[env_ids] = 0.0
        self.demand[env_ids] = 0.0
        self.satisfied_demand[env_ids] = 0.0
        self.current_step[env_ids] = 0
        self.done[env_ids] = False
        return self._observation()[env_ids]

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Advance all environments one step.

        Returns
        -------
        obs : (num_envs, num_firms, 3)
        rewards : (num_envs, num_firms, 1)
        dones : (num_envs,)
        infos : list of dicts, one per env
        """
        if np.any(self.done):
            raise RuntimeError("step() called on done environments; reset first")

        actions = self._clip_actions(actions)
        n, m = self.num_envs, self.num_firms
        prices = np.asarray(self.config.prices, dtype=np.float32)

        # Receive inbound shipments.
        inbound = self.pipeline.copy()
        self.inventory += inbound

        # Generate demand: external Poisson for firm 0, downstream orders for others.
        demand = np.zeros((n, m), dtype=np.float32)
        demand[:, 0] = np.array(
            [rng.poisson(self.config.poisson_lambda) for rng in self.rngs],
            dtype=np.float32,
        )
        if m > 1:
            demand[:, 1:] = actions[:, :-1]

        # Satisfied demand is bounded by inventory; leftover stays in inventory.
        satisfied = np.minimum(demand, self.inventory)
        self.inventory -= satisfied

        # Pipeline update: shipments sent downstream arrive next step; upstream order.
        next_pipeline = np.zeros((n, m), dtype=np.float32)
        if m > 1:
            next_pipeline[:, :-1] = satisfied[:, 1:]
        next_pipeline[:, -1] = actions[:, -1]

        # Rewards.
        revenue = prices * satisfied
        purchase_cost = np.zeros((n, m), dtype=np.float32)
        if m > 1:
            purchase_cost[:, :-1] = prices[1:] * actions[:, :-1]
        holding_cost = self.config.holding_cost * self.inventory
        lost_sales = self.config.lost_sales_cost * np.maximum(demand - satisfied, 0.0)
        rewards = revenue - purchase_cost - holding_cost - lost_sales

        self.pipeline = next_pipeline
        self.last_orders = actions
        self.demand = demand
        self.satisfied_demand = satisfied
        self.current_step += 1
        dones = self.current_step >= self.max_steps
        self.done = dones.copy()

        infos = [
            {
                "actions": actions[i].copy(),
                "demand": demand[i].copy(),
                "satisfied_demand": satisfied[i].copy(),
                "inventory": self.inventory[i].copy(),
                "inbound": inbound[i].copy(),
                "pipeline": self.pipeline[i].copy(),
            }
            for i in range(n)
        ]

        obs = self._observation()
        # Auto-reset finished environments so the next step starts a new episode.
        done_ids = np.nonzero(dones)[0]
        if done_ids.size > 0:
            self._reset_envs(done_ids)
        return obs, rewards.reshape(n, m, 1), dones, infos
