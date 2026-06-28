from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import load_config, make_env_config
from .dqn import DQNAgent
from .env import BeerGameEnv
from .experiments import DISPLAY_NAMES, evaluate_policy, setup_chinese_font
from .policies import build_policy
from .run_baselines import DQNPolicy, build_agent

# matplotlib 必须在 setup_chinese_font 之前导入并设置后端
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def evaluate_and_record(env: BeerGameEnv, policy, firm_id: int, seed: int, background_policy: str = "random"):
    """跑一个 episode 并记录详细行为序列。"""
    result = evaluate_policy(
        env,
        policy,
        firm_id,
        episodes=1,
        seed=seed,
        background_policy=background_policy,
    )
    history = result["histories"]
    # evaluate_policy 返回的是 list of lists，这里只有一个 episode
    return {k: np.asarray(v[0], dtype=np.float32) for k, v in history.items()}


def plot_behavior_comparison(
    records: dict[str, dict],
    output_path: str | Path,
    title: str = "最终策略订购行为对比",
):
    setup_chinese_font()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()
    metrics = [
        ("orders", "订货量", "订单数量"),
        ("inventory", "库存", "库存量"),
        ("demand", "需求", "需求数量"),
        ("rewards", "即时奖励", "奖励"),
    ]
    colors = {"random": "#9aa0a6", "base_stock": "#6f7782", "dueling_double_dqn": "#e15759"}

    for ax, (key, ylabel, subplot_title) in zip(axes, metrics):
        for name, record in records.items():
            steps = np.arange(len(record[key]))
            ax.plot(
                steps,
                record[key],
                label=DISPLAY_NAMES.get(name, name),
                color=colors.get(name, "#4e79a7"),
                linewidth=1.8,
                alpha=0.9,
            )
        ax.set_ylabel(ylabel)
        ax.set_title(subplot_title)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    for ax in axes[-2:]:
        ax.set_xlabel("时间步")

    fig.suptitle(title, fontsize=15, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=220)
    plt.close()
    print(f"saved behavior comparison to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot final policy ordering behavior over one episode.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--seed", type=int, default=123, help="用于生成对比 episode 的 seed")
    parser.add_argument("--skip-model-check", action="store_true", help="不加载 DQN 模型，只画规则策略")
    args = parser.parse_args()

    setup_chinese_font()
    cfg = load_config(args.config)
    env = BeerGameEnv(make_env_config(cfg))
    firm_id = int(cfg["experiment"].get("firm_id", 1))
    target_inventory = int(cfg["baselines"].get("base_stock_target", env.config.initial_inventory))
    model_dir = Path(cfg["experiment"].get("model_dir", "models/baselines"))
    figure_dir = Path("figures/policy_behavior")
    figure_dir.mkdir(parents=True, exist_ok=True)

    records = {}

    # random
    random_policy = build_policy("random", env.config.max_order, seed=args.seed)
    records["random"] = evaluate_and_record(env, random_policy, firm_id, args.seed, background_policy="random")

    # base_stock
    base_stock_policy = build_policy("base_stock", env.config.max_order, seed=args.seed, target_inventory=target_inventory)
    records["base_stock"] = evaluate_and_record(env, base_stock_policy, firm_id, args.seed, background_policy="random")

    # Dueling Double DQN
    model_path = model_dir / f"dueling_double_dqn_seed_{cfg['env'].get('seed', 42)}_firm_{firm_id}_tplus1.pt"
    if not args.skip_model_check and model_path.exists():
        agent = build_agent(
            env,
            cfg,
            {"name": "dueling_double_dqn", "network_type": "dueling", "double_dqn": True},
            firm_id,
            int(cfg["env"].get("seed", 42)),
        )
        agent.load(model_path)
        records["dueling_double_dqn"] = evaluate_and_record(
            env,
            DQNPolicy(agent),
            firm_id,
            args.seed,
            background_policy="random",
        )
    else:
        print(f"warning: {model_path} not found, skipping DQN behavior plot")

    output_path = figure_dir / "behavior_comparison.png"
    plot_behavior_comparison(records, output_path)

    # 同时保存原始数据便于报告使用
    data_path = figure_dir / "behavior_records.json"
    serializable = {
        name: {k: v.tolist() for k, v in record.items()}
        for name, record in records.items()
    }
    with data_path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"saved behavior records to {data_path}")


if __name__ == "__main__":
    main()
