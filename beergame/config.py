from __future__ import annotations

import json
from pathlib import Path

from .env import BeerGameConfig


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def make_env_config(raw: dict) -> BeerGameConfig:
    env = raw.get("env", raw)
    return BeerGameConfig(
        num_firms=int(env.get("num_firms", 3)),
        prices=tuple(float(x) for x in env.get("prices", [10, 9, 8])),
        holding_cost=float(env.get("holding_cost", 0.5)),
        lost_sales_cost=float(env.get("lost_sales_cost", 2.0)),
        initial_inventory=int(env.get("initial_inventory", 100)),
        poisson_lambda=float(env.get("poisson_lambda", 10)),
        max_steps=int(env.get("max_steps", 100)),
        max_order=int(env.get("max_order", 20)),
        seed=env.get("seed", 42),
    )

