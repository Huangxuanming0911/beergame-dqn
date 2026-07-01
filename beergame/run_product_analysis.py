from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import load_config, make_env_config
from .env import BeerGameConfig, BeerGameEnv
from .experiments import setup_chinese_font


POLICY_LABELS = {
    "random": "随机订货",
    "base_stock": "库存补足",
    "random_all": "全随机",
    "base_stock_all": "全库存补足",
    "single_agent_ddqn": "单智能体 DDQN",
    "multiagent_ddqn": "多智能体 DDQN",
}


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)


def _price_sequence(num_firms: int, profile: dict[str, Any]) -> tuple[float, ...]:
    start = float(profile.get("start", 10.0))
    step = float(profile.get("step", -0.6))
    floor = float(profile.get("floor", 6.0))
    return tuple(max(floor, start + step * i) for i in range(num_firms))


def _scenario_config(base: BeerGameConfig, product_cfg: dict[str, Any], num_firms: int, demand_lambda: float) -> BeerGameConfig:
    prices = _price_sequence(num_firms, product_cfg.get("price_profile", {}))
    cost_profile = product_cfg.get("cost_profile", {})
    return replace(
        base,
        num_firms=num_firms,
        prices=prices,
        poisson_lambda=float(demand_lambda),
        holding_cost=float(cost_profile.get("holding_cost", base.holding_cost)),
        lost_sales_cost=float(cost_profile.get("lost_sales_cost", base.lost_sales_cost)),
    )


def _act(policy_name: str, env: BeerGameEnv, state: np.ndarray, rng: np.random.Generator, target_inventory: int) -> np.ndarray:
    if policy_name == "random":
        return rng.integers(0, env.config.max_order + 1, size=env.num_firms).astype(np.float32)
    if policy_name == "base_stock":
        inventory = state[:, 2]
        return np.clip(np.rint(target_inventory - inventory), 0, env.config.max_order).astype(np.float32)
    raise ValueError(f"unknown product-analysis policy: {policy_name}")


def _empty_episode_totals(num_firms: int) -> dict[str, np.ndarray]:
    return {
        "reward": np.zeros(num_firms, dtype=np.float64),
        "revenue": np.zeros(num_firms, dtype=np.float64),
        "purchase_cost": np.zeros(num_firms, dtype=np.float64),
        "holding_cost": np.zeros(num_firms, dtype=np.float64),
        "lost_sales_penalty": np.zeros(num_firms, dtype=np.float64),
        "demand": np.zeros(num_firms, dtype=np.float64),
        "satisfied": np.zeros(num_firms, dtype=np.float64),
        "stockout": np.zeros(num_firms, dtype=np.float64),
        "inventory": np.zeros(num_firms, dtype=np.float64),
        "orders": np.zeros(num_firms, dtype=np.float64),
    }


def evaluate_policy_metrics(
    env_config: BeerGameConfig,
    policy_name: str,
    seeds: list[int],
    episodes_per_seed: int,
    target_inventory: int,
) -> dict[str, Any]:
    episode_rows: list[dict[str, Any]] = []
    order_series: list[np.ndarray] = []
    external_demand_series: list[float] = []

    for seed in seeds:
        env = BeerGameEnv(env_config)
        rng = np.random.default_rng(seed)
        for episode in range(episodes_per_seed):
            state = env.reset(seed=seed + episode)
            totals = _empty_episode_totals(env.num_firms)
            steps = 0
            done = False
            while not done:
                actions = _act(policy_name, env, state, rng, target_inventory)
                state, rewards, done, info = env.step(actions)
                components = info["reward_components"]
                totals["reward"] += rewards[:, 0]
                totals["revenue"] += components["revenue"]
                totals["purchase_cost"] += components["purchase_cost"]
                totals["holding_cost"] += components["holding_cost"]
                totals["lost_sales_penalty"] += components["lost_sales_penalty"]
                totals["demand"] += info["demand"]
                totals["satisfied"] += info["satisfied_demand"]
                totals["stockout"] += info["stockout"]
                totals["inventory"] += info["inventory"]
                totals["orders"] += info["actions"]
                order_series.append(info["actions"].astype(np.float64))
                external_demand_series.append(float(info["demand"][0]))
                steps += 1

            row = {key: value.copy() for key, value in totals.items()}
            row["seed"] = seed
            row["episode"] = episode
            row["steps"] = steps
            episode_rows.append(row)

    rewards = np.vstack([row["reward"] for row in episode_rows])
    revenue = np.vstack([row["revenue"] for row in episode_rows])
    purchase = np.vstack([row["purchase_cost"] for row in episode_rows])
    holding = np.vstack([row["holding_cost"] for row in episode_rows])
    lost = np.vstack([row["lost_sales_penalty"] for row in episode_rows])
    demand = np.vstack([row["demand"] for row in episode_rows])
    satisfied = np.vstack([row["satisfied"] for row in episode_rows])
    stockout = np.vstack([row["stockout"] for row in episode_rows])
    inventory = np.vstack([row["inventory"] for row in episode_rows])
    orders = np.vstack([row["orders"] for row in episode_rows])
    order_steps = np.vstack(order_series)
    external_demand = np.asarray(external_demand_series, dtype=np.float64)
    demand_variance = float(np.var(external_demand))
    order_variance = np.var(order_steps, axis=0)
    firm_mean_rewards = rewards.mean(axis=0)
    reward_scale = float(np.mean(np.abs(firm_mean_rewards)) + 1e-9)

    summary = {
        "policy": policy_name,
        "num_firms": env_config.num_firms,
        "poisson_lambda": env_config.poisson_lambda,
        "episodes": len(episode_rows),
        "total_chain_mean_reward": float(rewards.sum(axis=1).mean()),
        "total_chain_std_reward": float(rewards.sum(axis=1).std()),
        "firm_mean_rewards": firm_mean_rewards,
        "firm_std_rewards": rewards.std(axis=0),
        "service_level_by_firm": np.divide(satisfied.sum(axis=0), demand.sum(axis=0), out=np.ones(env_config.num_firms), where=demand.sum(axis=0) > 0),
        "stockout_rate_by_firm": np.divide(stockout.sum(axis=0), demand.sum(axis=0), out=np.zeros(env_config.num_firms), where=demand.sum(axis=0) > 0),
        "avg_inventory_by_firm": inventory.mean(axis=0) / env_config.max_steps,
        "avg_order_by_firm": orders.mean(axis=0) / env_config.max_steps,
        "order_std_by_firm": order_steps.std(axis=0),
        "bullwhip_ratio_by_firm": order_variance / demand_variance if demand_variance > 0 else np.zeros(env_config.num_firms),
        "reward_imbalance_index": float(firm_mean_rewards.std() / reward_scale),
        "reward_components": {
            "revenue": revenue.mean(axis=0),
            "purchase_cost": purchase.mean(axis=0),
            "holding_cost": holding.mean(axis=0),
            "lost_sales_penalty": lost.mean(axis=0),
            "final_reward": firm_mean_rewards,
        },
    }
    return summary


def _plot_chain_map(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.axis("off")
    xs = np.linspace(0.08, 0.92, 5)
    labels = ["外部\n需求", "企业0\n零售端", "企业1\n批发端", "企业2\n分销端", "上游\n供给"]
    colors = ["#f2f4f7", "#d9e8ff", "#dff3e3", "#fff0cc", "#f2f4f7"]
    for x, label, color in zip(xs, labels, colors):
        ax.text(
            x,
            0.55,
            label,
            ha="center",
            va="center",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.55", "facecolor": color, "edgecolor": "#5c6470", "linewidth": 1.2},
        )
    for start, end in zip(xs[:-1], xs[1:]):
        ax.annotate("", xy=(end - 0.06, 0.58), xytext=(start + 0.06, 0.58), arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "#355070"})
        ax.annotate("", xy=(start + 0.06, 0.42), xytext=(end - 0.06, 0.42), arrowprops={"arrowstyle": "->", "lw": 1.3, "color": "#c65d3a"})
    ax.text(0.5, 0.78, "货物 / 已满足需求沿下游流动", ha="center", fontsize=11, color="#355070")
    ax.text(0.5, 0.22, "订单与采购成本向上游传递；未满足需求形成缺货惩罚", ha="center", fontsize=11, color="#8a3f2b")
    ax.set_title("啤酒游戏企业链路图", fontsize=16, pad=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_reward_decomposition(decomposition: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_stock = decomposition.get("base_stock") or next(iter(decomposition.values()))
    comp = base_stock["reward_components"]
    firms = np.arange(len(comp["final_reward"]))
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.bar(firms, comp["revenue"], label="销售收入", color="#4e79a7")
    ax.bar(firms, -np.asarray(comp["purchase_cost"]), label="采购成本", color="#f28e2b")
    ax.bar(firms, -np.asarray(comp["holding_cost"]), bottom=-np.asarray(comp["purchase_cost"]), label="库存持有成本", color="#e15759")
    neg_base = -np.asarray(comp["purchase_cost"]) - np.asarray(comp["holding_cost"])
    ax.bar(firms, -np.asarray(comp["lost_sales_penalty"]), bottom=neg_base, label="缺货惩罚", color="#b07aa1")
    ax.plot(firms, comp["final_reward"], marker="o", color="#222222", linewidth=2, label="最终 reward")
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_xticks(firms)
    ax.set_xticklabels([f"企业 {i}" for i in firms])
    ax.set_ylabel("平均单局数值")
    ax.set_title("各企业 reward 分解")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_chain_length_heatmap(chain_summary: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [item for item in chain_summary["scenarios"] if item["policy"] == "base_stock"]
    lengths = [int(item["num_firms"]) for item in rows]
    max_len = max(lengths)
    matrix = np.full((len(rows), max_len), np.nan)
    for r, item in enumerate(rows):
        rewards = np.asarray(item["firm_mean_rewards"], dtype=np.float64)
        matrix[r, : len(rewards)] = rewards
    fig, ax = plt.subplots(figsize=(10, 4.8))
    image = ax.imshow(matrix, aspect="auto", cmap="RdYlGn")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([f"{length} 个企业" for length in lengths])
    ax.set_xticks(np.arange(max_len))
    ax.set_xticklabels([f"企业 {i}" for i in range(max_len)])
    ax.set_title("长链路下不同位置的 reward")
    ax.set_xlabel("企业位置")
    ax.set_ylabel("链路长度")
    fig.colorbar(image, ax=ax, label="平均 reward")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_demand_robustness(demand_summary: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for policy in sorted({item["policy"] for item in demand_summary["scenarios"]}):
        rows = [item for item in demand_summary["scenarios"] if item["policy"] == policy]
        rows.sort(key=lambda item: item["poisson_lambda"])
        ax.plot(
            [item["poisson_lambda"] for item in rows],
            [item["total_chain_mean_reward"] for item in rows],
            marker="o",
            linewidth=2,
            label=POLICY_LABELS.get(policy, policy),
        )
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_xlabel("外部需求强度 lambda")
    ax.set_ylabel("全链路平均 reward")
    ax.set_title("不同需求强度下的策略稳健性")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_bullwhip(summary: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        item
        for item in summary["scenarios"]
        if int(item["num_firms"]) == summary["default_num_firms"] and float(item["poisson_lambda"]) == summary["default_lambda"]
    ]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    width = 0.36
    x = np.arange(summary["default_num_firms"])
    for idx, item in enumerate(rows):
        offset = (idx - (len(rows) - 1) / 2) * width
        ax.bar(x + offset, item["bullwhip_ratio_by_firm"], width=width, label=POLICY_LABELS.get(item["policy"], item["policy"]), alpha=0.88)
    ax.axhline(1, color="#333333", linewidth=1, linestyle="--", label="外部需求方差")
    ax.set_xticks(x)
    ax.set_xticklabels([f"企业 {i}" for i in x])
    ax.set_ylabel("订单方差 / 需求方差")
    ax.set_title("牛鞭效应对比")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_local_global_tradeoff(summary_path: Path, output_path: Path) -> bool:
    if not summary_path.exists():
        return False
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    methods = [name for name in ["random_all", "base_stock_all", "single_agent_ddqn", "multiagent_ddqn"] if name in summary]
    if not methods:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(methods))
    firm1 = [summary[name].get("firm_1_mean_reward", 0.0) for name in methods]
    total = [summary[name].get("total_chain_mean_reward", 0.0) for name in methods]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.bar(x - 0.18, firm1, width=0.36, label="目标企业 reward", color="#4e79a7")
    ax.bar(x + 0.18, total, width=0.36, label="全链路 reward", color="#e15759")
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_LABELS.get(name, name) for name in methods], rotation=15, ha="right")
    ax.set_ylabel("平均单局 reward")
    ax.set_title("局部最优与全链路结果对比")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return True


def run_product_analysis(config_path: str | Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    base_env = make_env_config(cfg)
    product_cfg = cfg.get("product_analysis", {})
    chain_lengths = [int(x) for x in product_cfg.get("chain_lengths", [3, 5, 7])]
    demand_lambdas = [float(x) for x in product_cfg.get("demand_lambdas", [8, 10, 12])]
    policies = list(product_cfg.get("policies", ["random", "base_stock"]))
    seeds = [int(x) for x in product_cfg.get("seeds", [42, 123, 456])]
    eval_episodes = int(product_cfg.get("eval_episodes", cfg.get("experiment", {}).get("eval_episodes", 20)))
    target_inventory = int(cfg.get("baselines", {}).get("base_stock_target", base_env.initial_inventory))
    default_length = int(product_cfg.get("default_num_firms", base_env.num_firms))
    default_lambda = float(product_cfg.get("default_lambda", base_env.poisson_lambda))
    output_dir = Path(product_cfg.get("output_dir", "results/product"))
    figure_dir = Path(product_cfg.get("figure_dir", "figures/product"))

    policy_metrics: list[dict[str, Any]] = []
    for num_firms in sorted(set(chain_lengths + [default_length])):
        for demand_lambda in sorted(set(demand_lambdas + [default_lambda])):
            scenario_env = _scenario_config(base_env, product_cfg, num_firms, demand_lambda)
            for policy in policies:
                policy_metrics.append(
                    evaluate_policy_metrics(
                        scenario_env,
                        policy,
                        seeds=seeds,
                        episodes_per_seed=eval_episodes,
                        target_inventory=target_inventory,
                    )
                )

    chain_scenarios = [
        item
        for item in policy_metrics
        if float(item["poisson_lambda"]) == default_lambda and int(item["num_firms"]) in chain_lengths
    ]
    demand_scenarios = [
        item
        for item in policy_metrics
        if int(item["num_firms"]) == default_length and float(item["poisson_lambda"]) in demand_lambdas
    ]
    decomposition = {
        item["policy"]: item
        for item in policy_metrics
        if int(item["num_firms"]) == default_length and float(item["poisson_lambda"]) == default_lambda
    }

    policy_summary = {
        "default_num_firms": default_length,
        "default_lambda": default_lambda,
        "seeds": seeds,
        "eval_episodes_per_seed": eval_episodes,
        "scenarios": policy_metrics,
    }
    chain_summary = {"default_lambda": default_lambda, "scenarios": chain_scenarios}
    demand_summary = {"default_num_firms": default_length, "scenarios": demand_scenarios}

    _write_json(output_dir / "policy_metrics_summary.json", policy_summary)
    _write_json(output_dir / "chain_length_summary.json", chain_summary)
    _write_json(output_dir / "demand_robustness_summary.json", demand_summary)
    _write_json(output_dir / "reward_decomposition.json", decomposition)

    setup_chinese_font()
    _plot_chain_map(figure_dir / "chain_map.png")
    _plot_reward_decomposition(decomposition, figure_dir / "reward_decomposition.png")
    _plot_chain_length_heatmap(chain_summary, figure_dir / "chain_length_heatmap.png")
    _plot_demand_robustness(demand_summary, figure_dir / "demand_robustness.png")
    _plot_bullwhip(policy_summary, figure_dir / "bullwhip_comparison.png")
    _plot_local_global_tradeoff(Path("results/multiagent/multiagent_summary.json"), figure_dir / "local_global_tradeoff.png")

    return policy_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate product-style Beer Game analysis assets.")
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()
    summary = run_product_analysis(args.config)
    print(json.dumps({"scenarios": len(summary["scenarios"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
