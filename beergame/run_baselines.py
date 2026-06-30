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
    train_ppo_best,
    train_sac,
)
from .policies import build_policy
from .ppo import PPOAgent
from .sac_discrete import DiscreteSACAgent


class DQNPolicy:
    def __init__(self, agent: DQNAgent):
        self.agent = agent

    def act(self, state, firm_id: int) -> int:
        return self.agent.act(state[firm_id], epsilon=0.0)


class PPOPolicy:
    def __init__(self, agent: PPOAgent):
        self.agent = agent

    def reset(self):
        self.agent.reset_history()

    def act(self, state, firm_id: int) -> int:
        return self.agent.eval_act(state[firm_id], use_ema=self.agent.use_ema)


class SACPolicy:
    def __init__(self, agent: DiscreteSACAgent):
        self.agent = agent

    def act(self, state, firm_id: int) -> int:
        with torch.no_grad():
            state_t = torch.FloatTensor(state[firm_id]).unsqueeze(0).to(self.agent.device)
            q1 = self.agent.q1(state_t)
            q2 = self.agent.q2(state_t)
            q = torch.min(q1, q2)
            action = q.argmax(dim=-1)
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
    episodes = int(ppo_cfg.get("episodes", 300))
    rollout_episodes = int(ppo_cfg.get("rollout_episodes", 4))
    total_updates = max(1, episodes // rollout_episodes)
    state_history_len = int(ppo_cfg.get("state_history_len", 1))
    centralized_critic = bool(ppo_cfg.get("centralized_critic", False))
    critic_state_dim = 3 * env.num_firms if centralized_critic else 3 * state_history_len
    state_dim = 3 * state_history_len
    return PPOAgent(
        state_dim=state_dim,
        action_dim=env.config.max_order + 1,
        firm_id=firm_id,
        lr=float(ppo_cfg.get("learning_rate", 1e-4)),
        gamma=float(ppo_cfg.get("gamma", 0.99)),
        gae_lambda=float(ppo_cfg.get("gae_lambda", 0.95)),
        clip_epsilon=float(ppo_cfg.get("clip_epsilon", 0.2)),
        value_coef=float(ppo_cfg.get("value_coef", 0.5)),
        entropy_coef=float(ppo_cfg.get("entropy_coef", 0.05)),
        hidden_size=int(ppo_cfg.get("hidden_size", 64)),
        update_epochs=int(ppo_cfg.get("update_epochs", 10)),
        batch_size=int(ppo_cfg.get("batch_size", 256)),
        max_grad_norm=float(ppo_cfg.get("max_grad_norm", 0.5)),
        target_kl=float(ppo_cfg.get("target_kl", 0.015)),
        separate_actor_critic=bool(ppo_cfg.get("separate_actor_critic", False)),
        rollout_episodes=rollout_episodes,
        use_reward_norm=bool(ppo_cfg.get("use_reward_norm", True)),
        use_state_norm=bool(ppo_cfg.get("use_state_norm", True)),
        use_value_clip=bool(ppo_cfg.get("use_value_clip", True)),
        use_lr_decay=bool(ppo_cfg.get("use_lr_decay", True)),
        use_entropy_decay=bool(ppo_cfg.get("use_entropy_decay", True)),
        state_history_len=state_history_len,
        use_ema=bool(ppo_cfg.get("use_ema", False)),
        ema_tau=float(ppo_cfg.get("ema_tau", 0.005)),
        centralized_critic=centralized_critic,
        critic_state_dim=critic_state_dim,
        total_updates=total_updates,
        activation=str(ppo_cfg.get("activation", "relu")),
        use_layer_norm=bool(ppo_cfg.get("use_layer_norm", False)),
    )


def build_sac_agent(env: BeerGameEnv, cfg: dict, firm_id: int, seed: int) -> DiscreteSACAgent:
    sac_cfg = cfg.get("sac", {})
    return DiscreteSACAgent(
        state_dim=3,
        action_dim=env.config.max_order + 1,
        firm_id=firm_id,
        lr=float(sac_cfg.get("learning_rate", 3e-4)),
        gamma=float(sac_cfg.get("gamma", 0.99)),
        tau=float(sac_cfg.get("tau", 0.005)),
        alpha=float(sac_cfg.get("alpha", 0.2)),
        hidden_size=int(sac_cfg.get("hidden_size", 64)),
        buffer_size=int(sac_cfg.get("buffer_size", 100000)),
        batch_size=int(sac_cfg.get("batch_size", 64)),
        update_every=int(sac_cfg.get("update_every", 1)),
        reward_scale=float(sac_cfg.get("reward_scale", 1.0)),
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
    parser.add_argument("--only", default=None, help="Comma-separated list of algorithm names to run (e.g., ppo,dueling_double_dqn).")
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
    if args.only:
        only_set = set(args.only.split(","))
        algorithms = [a for a in algorithms if a["name"] in only_set]
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
        algo_type = algorithm.get("type", "dqn").lower()
        print(f"\n=== {name} ===")
        all_training_scores = []
        all_eval_scores = []
        seed_eval_means = []

        for seed in train_seeds:
            print(f"--- seed={seed} ---")
            if algo_type == "ppo":
                agent = build_ppo_agent(env, cfg, firm_id, seed)
                policy_wrapper = PPOPolicy(agent)
                train_cfg = {**cfg.get("ppo", {}), "seed": seed}
                train_fn = train_ppo_best
            elif algo_type == "sac":
                agent = build_sac_agent(env, cfg, firm_id, seed)
                policy_wrapper = SACPolicy(agent)
                train_cfg = {**cfg.get("sac", {}), "seed": seed}
                train_fn = train_sac
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
