from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import load_config, make_env_config
from .dqn import DQNAgent
from .env import BeerGameEnv
from .experiments import (
    evaluate_policy,
    plot_baseline_comparison,
    plot_training,
    train_dqn,
    train_ppo,
)
from .policies import build_policy
from .ppo import PPOAgent


class DQNPolicy:
    def __init__(self, agent: DQNAgent):
        self.agent = agent

    def act(self, state, firm_id: int) -> int:
        return self.agent.act(state[firm_id], epsilon=0.0)


class PPOPolicy:
    def __init__(self, agent: PPOAgent):
        self.agent = agent

    def act(self, state, firm_id: int) -> int:
        with torch.no_grad():
            state_t = torch.FloatTensor(state[firm_id]).unsqueeze(0).to(self.agent.device)
            logits, _ = self.agent.net.forward(state_t)
            action = logits.argmax(dim=-1)
        return int(action.item())


def build_agent(env: BeerGameEnv, cfg: dict, algorithm: dict, firm_id: int, seed: int) -> DQNAgent:
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
        seed=seed,
    )


def build_ppo_agent(env: BeerGameEnv, cfg: dict, firm_id: int, seed: int) -> PPOAgent:
    ppo_cfg = cfg.get("ppo", {})
    return PPOAgent(
        state_dim=3,
        action_dim=env.config.max_order + 1,
        firm_id=firm_id,
        lr=float(ppo_cfg.get("learning_rate", 1e-4)),
        gamma=float(ppo_cfg.get("gamma", 0.99)),
        gae_lambda=float(ppo_cfg.get("gae_lambda", 0.95)),
        clip_epsilon=float(ppo_cfg.get("clip_epsilon", 0.2)),
        value_coef=float(ppo_cfg.get("value_coef", 0.5)),
        entropy_coef=float(ppo_cfg.get("entropy_coef", 0.05)),
        hidden_size=int(ppo_cfg.get("hidden_size", 64)),
        update_epochs=int(ppo_cfg.get("update_epochs", 4)),
        batch_size=int(ppo_cfg.get("batch_size", 64)),
        max_grad_norm=float(ppo_cfg.get("max_grad_norm", 0.5)),
    )


def evaluate_over_seeds(env: BeerGameEnv, policy, firm_id: int, eval_episodes: int, seeds: list[int]):
    seed_scores = []
    for seed in seeds:
        result = evaluate_policy(env, policy, firm_id, eval_episodes, seed=seed)
        seed_scores.append(result["scores"])
    return {"scores": np.concatenate(seed_scores), "seed_scores": seed_scores}


def evaluate_rule_policy_over_seeds(
    env: BeerGameEnv,
    policy_name: str,
    firm_id: int,
    eval_episodes: int,
    seeds: list[int],
    target_inventory: int,
):
    seed_scores = []
    seed_means = []
    for seed in seeds:
        policy = build_policy(policy_name, env.config.max_order, seed=seed, target_inventory=target_inventory)
        result = evaluate_policy(env, policy, firm_id, eval_episodes, seed=seed)
        seed_scores.append(result["scores"])
        seed_means.append(float(np.mean(result["scores"])))
    return {"scores": np.concatenate(seed_scores), "seed_mean_rewards": seed_means}


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

    algorithms = cfg.get(
        "algorithms",
        [{"name": "dqn", "network_type": "standard", "double_dqn": False}],
    )
    train_seeds = [int(seed) for seed in cfg["experiment"].get("train_seeds", [int(cfg["env"].get("seed", 42))])]
    eval_episodes = int(cfg["experiment"].get("eval_episodes", 20))

    # 规则策略不需要训练，但也在同一组seed上评估，便于和学习算法公平比较。
    results = {}
    target = int(cfg["baselines"].get("base_stock_target", env.config.initial_inventory))
    for policy_name in ["random", "base_stock"]:
        results[policy_name] = evaluate_rule_policy_over_seeds(
            env,
            policy_name,
            firm_id,
            eval_episodes,
            train_seeds,
            target,
        )

    for algorithm in algorithms:
        name = algorithm["name"]
        is_ppo = algorithm.get("type", "dqn").lower() == "ppo"
        print(f"\n=== {name} ===")
        all_training_scores = []
        all_eval_scores = []
        seed_eval_means = []

        for seed in train_seeds:
            print(f"--- seed={seed} ---")
            if is_ppo:
                agent = build_ppo_agent(env, cfg, firm_id, seed)
                policy_wrapper = PPOPolicy(agent)
                train_cfg = {**cfg.get("ppo", {}), "seed": seed}
                train_fn = train_ppo
            else:
                agent = build_agent(env, cfg, algorithm, firm_id, seed)
                policy_wrapper = DQNPolicy(agent)
                train_cfg = {**cfg["dqn"], "seed": seed}
                train_fn = train_dqn

            model_path = model_dir / f"{name}_seed_{seed}_firm_{firm_id}_tplus1.pt"
            if args.skip_train and model_path.exists():
                agent.load(model_path)
            else:
                scores = train_fn(env, agent, train_cfg)
                agent.save(model_path)
                np.save(output_dir / f"{name}_seed_{seed}_training_scores.npy", scores)
            if (output_dir / f"{name}_seed_{seed}_training_scores.npy").exists():
                all_training_scores.append(np.load(output_dir / f"{name}_seed_{seed}_training_scores.npy"))

            eval_result = evaluate_policy(
                env,
                policy_wrapper,
                firm_id,
                eval_episodes,
                seed=seed,
            )
            all_eval_scores.append(eval_result["scores"])
            seed_eval_means.append(float(np.mean(eval_result["scores"])))

        stacked_training_scores = np.vstack(all_training_scores)
        np.save(output_dir / f"{name}_training_scores.npy", stacked_training_scores)
        plot_training(
            stacked_training_scores,
            figure_dir / f"{name}_training_rewards.png",
            window=int(cfg["dqn"].get("plot_window", 50)),
        )
        results[name] = {
            "scores": np.concatenate(all_eval_scores),
            "seed_mean_rewards": seed_eval_means,
        }

    summary = {
        name: {
            "mean_reward": float(np.mean(result["scores"])),
            "std_reward": float(np.std(result["scores"])),
            "episodes": int(len(result["scores"])),
            "train_seeds": train_seeds,
            "eval_episodes_per_seed": eval_episodes,
            **(
                {"seed_mean_rewards": result["seed_mean_rewards"]}
                if "seed_mean_rewards" in result
                else {}
            ),
        }
        for name, result in results.items()
    }
    with (output_dir / "baseline_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_baseline_comparison(results, figure_dir / "baseline_comparison.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
