from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import load_config, make_env_config
from .dqn import DQNAgent
from .env import BeerGameEnv
from .experiments import evaluate_policy, plot_background_policy_comparison, plot_training, train_dqn
from .run_baselines import DQNPolicy, build_agent


BACKGROUND_ALGORITHM = {
    "name": "dueling_double_dqn",
    "network_type": "dueling",
    "double_dqn": True,
}

BACKGROUND_POLICIES = ["random", "base_stock"]


def result_name(background_policy: str) -> str:
    return f"{background_policy}_background"


def summarize(scores: np.ndarray, eval_episodes: int, seed: int, background_policy: str, model_path: Path) -> dict:
    return {
        "mean_reward": float(np.mean(scores)),
        "std_reward": float(np.std(scores)),
        "episodes": int(eval_episodes),
        "seed": int(seed),
        "background_policy": background_policy,
        "model_path": str(model_path),
    }


def train_and_evaluate_background(
    cfg: dict,
    background_policy: str,
    firm_id: int,
    seed: int,
    eval_episodes: int,
    model_dir: Path,
    output_dir: Path,
    figure_dir: Path,
    target_inventory: int,
    skip_train: bool,
):
    env = BeerGameEnv(make_env_config(cfg))
    agent: DQNAgent = build_agent(env, cfg, BACKGROUND_ALGORITHM, firm_id, seed)
    name = result_name(background_policy)
    model_path = model_dir / f"dueling_double_dqn_{name}_seed_{seed}_firm_{firm_id}_tplus1.pt"
    training_path = output_dir / f"dueling_double_dqn_{name}_seed_{seed}_training_scores.npy"
    training_figure_path = figure_dir / f"dueling_double_dqn_{name}_training_rewards.png"

    if skip_train and model_path.exists():
        agent.load(model_path)
        if training_path.exists():
            scores = np.load(training_path)
            plot_training(scores, training_figure_path, window=int(cfg["dqn"].get("plot_window", 50)))
    else:
        train_cfg = {
            **cfg["dqn"],
            "seed": seed,
            "background_policy": background_policy,
            "background_base_stock_target": target_inventory,
        }
        scores = train_dqn(env, agent, train_cfg)
        agent.save(model_path)
        np.save(training_path, scores)
        plot_training(scores, training_figure_path, window=int(cfg["dqn"].get("plot_window", 50)))

    eval_result = evaluate_policy(
        env,
        DQNPolicy(agent),
        firm_id,
        eval_episodes,
        seed=seed,
        background_policy=background_policy,
        background_target_inventory=target_inventory,
    )
    return eval_result, summarize(eval_result["scores"], eval_episodes, seed, background_policy, model_path)


def main():
    parser = argparse.ArgumentParser(description="Compare random and base-stock background policies.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    firm_id = int(cfg["experiment"].get("firm_id", 1))
    eval_episodes = int(cfg["experiment"].get("eval_episodes", 20))
    background_cfg = cfg.get("background_experiment", {})
    seed = int(background_cfg.get("seed", cfg["experiment"].get("demand_model_seed", 42)))
    model_dir = Path(background_cfg.get("model_dir", "models/background"))
    output_dir = Path(background_cfg.get("output_dir", "results/background"))
    figure_dir = Path(background_cfg.get("figure_dir", "figures/background"))
    target_inventory = int(cfg["baselines"].get("base_stock_target", 100))
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    summary = {}
    for background_policy in BACKGROUND_POLICIES:
        print(f"\n=== background policy: {background_policy} ===")
        eval_result, result_summary = train_and_evaluate_background(
            cfg,
            background_policy,
            firm_id,
            seed,
            eval_episodes,
            model_dir,
            output_dir,
            figure_dir,
            target_inventory,
            args.skip_train,
        )
        name = result_name(background_policy)
        results[name] = eval_result
        summary[name] = result_summary

    with (output_dir / "background_policy_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    plot_background_policy_comparison(results, figure_dir / "background_policy_comparison.png")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
