from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import load_config, make_env_config
from .dqn import DQNAgent
from .env import BeerGameEnv
from .experiments import (
    evaluate_policy,
    plot_baseline_comparison,
    plot_training,
    run_rule_baselines,
    train_dqn,
)


class DQNPolicy:
    def __init__(self, agent: DQNAgent):
        self.agent = agent

    def act(self, state, firm_id: int) -> int:
        return self.agent.act(state[firm_id], epsilon=0.0)


def build_agent(env: BeerGameEnv, cfg: dict, algorithm: dict, firm_id: int) -> DQNAgent:
    return DQNAgent(
        state_size=3,
        action_size=env.config.max_order + 1,
        firm_id=firm_id,
        hidden_size=int(cfg["dqn"].get("hidden_size", 64)),
        buffer_size=int(cfg["dqn"].get("buffer_size", 10000)),
        batch_size=int(cfg["dqn"].get("batch_size", 64)),
        gamma=float(cfg["dqn"].get("gamma", 0.99)),
        learning_rate=float(cfg["dqn"].get("learning_rate", 1e-3)),
        tau=float(cfg["dqn"].get("tau", 1e-3)),
        update_every=int(cfg["dqn"].get("update_every", 4)),
        network_type=algorithm.get("network_type", "standard"),
        double_dqn=bool(algorithm.get("double_dqn", False)),
        seed=cfg["env"].get("seed", 42),
    )


def main():
    parser = argparse.ArgumentParser(description="Run Beergame baselines with t+1 delivery.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env = BeerGameEnv(make_env_config(cfg))
    firm_id = int(cfg["experiment"].get("firm_id", 1))
    output_dir = Path(cfg["experiment"].get("output_dir", "results/baselines"))
    figure_dir = Path(cfg["experiment"].get("figure_dir", "figures/baselines"))
    model_dir = Path(cfg["experiment"].get("model_dir", "models/baselines"))
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    results = run_rule_baselines(env, cfg)
    algorithms = cfg.get(
        "algorithms",
        [{"name": "dqn", "network_type": "standard", "double_dqn": False}],
    )

    for algorithm in algorithms:
        name = algorithm["name"]
        print(f"\n=== {name} ===")
        agent = build_agent(env, cfg, algorithm, firm_id)
        model_path = model_dir / f"{name}_firm_{firm_id}_tplus1.pt"
        if args.skip_train and model_path.exists():
            agent.load(model_path)
        else:
            scores = train_dqn(env, agent, cfg["dqn"])
            agent.save(model_path)
            np.save(output_dir / f"{name}_training_scores.npy", scores)
            plot_training(
                scores,
                figure_dir / f"{name}_training_rewards.png",
                window=int(cfg["dqn"].get("plot_window", 50)),
            )

        results[name] = evaluate_policy(
            env,
            DQNPolicy(agent),
            firm_id,
            int(cfg["experiment"].get("eval_episodes", 20)),
            seed=cfg["env"].get("seed", 42),
        )

    summary = {
        name: {
            "mean_reward": float(np.mean(result["scores"])),
            "std_reward": float(np.std(result["scores"])),
            "episodes": int(len(result["scores"])),
        }
        for name, result in results.items()
    }
    with (output_dir / "baseline_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_baseline_comparison(results, figure_dir / "baseline_comparison.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
