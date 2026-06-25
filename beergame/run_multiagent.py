from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import load_config, make_env_config
from .dqn import DQNAgent
from .env import BeerGameEnv
from .experiments import plot_multiagent_comparison, plot_multiagent_eval_curve, plot_multiagent_training
from .policies import build_policy
from .run_baselines import DQNPolicy, build_agent


MULTIAGENT_ALGORITHM = {
    "name": "dueling_double_dqn",
    "network_type": "dueling",
    "double_dqn": True,
}


class MultiAgentPolicy:
    def __init__(self, agents: list[DQNAgent]):
        self.agents = agents

    def act_all(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(
            [agent.act(state[agent.firm_id], epsilon=0.0) for agent in self.agents],
            dtype=np.float32,
        )


def build_multiagent_agents(env: BeerGameEnv, cfg: dict, seed: int) -> list[DQNAgent]:
    return [
        build_agent(env, cfg, MULTIAGENT_ALGORITHM, firm_id, seed + firm_id)
        for firm_id in range(env.num_firms)
    ]


def train_multiagent(env: BeerGameEnv, agents: list[DQNAgent], cfg: dict, seed: int):
    scores = []
    eval_points = []
    eval_scores = []
    eps = float(cfg.get("eps_start", 1.0))
    eps_end = float(cfg.get("eps_end", 0.05))
    eps_decay = float(cfg.get("eps_decay", 0.99))
    episodes = int(cfg.get("episodes", 300))
    log_every = int(cfg.get("log_every", 50))
    eval_every = int(cfg.get("eval_every", 50))
    eval_episodes = int(cfg.get("eval_episodes", 20))

    for episode in range(1, episodes + 1):
        state = env.reset(seed=seed + episode)
        done = False
        episode_rewards = np.zeros(env.num_firms, dtype=np.float32)

        while not done:
            actions = np.asarray(
                [agent.act(state[agent.firm_id], epsilon=eps) for agent in agents],
                dtype=np.float32,
            )
            next_state, rewards, done, _ = env.step(actions)
            reward_vector = rewards[:, 0].astype(np.float32)
            for agent in agents:
                firm_id = agent.firm_id
                agent.step(
                    state[firm_id],
                    int(actions[firm_id]),
                    float(reward_vector[firm_id]),
                    next_state[firm_id],
                    done,
                )
            episode_rewards += reward_vector
            state = next_state

        eps = max(eps_end, eps_decay * eps)
        scores.append(episode_rewards.copy())
        if episode % log_every == 0:
            recent = np.asarray(scores[-log_every:], dtype=np.float32)
            print(
                f"episode={episode} "
                f"firm_avg={np.round(recent.mean(axis=0), 2).tolist()} "
                f"total_avg={recent.sum(axis=1).mean():.2f} "
                f"epsilon={eps:.3f}"
            )
        if eval_every > 0 and episode % eval_every == 0:
            eval_result = evaluate_policy_all(
                env,
                MultiAgentPolicy(agents),
                eval_episodes,
                seed + 10000 + episode,
            )
            eval_points.append(episode)
            eval_scores.append(eval_result.mean(axis=0))
    return (
        np.asarray(scores, dtype=np.float32),
        np.asarray(eval_points, dtype=np.int32),
        np.asarray(eval_scores, dtype=np.float32),
    )


def evaluate_policy_all(env: BeerGameEnv, policy, episodes: int, seed: int):
    scores = []
    for episode in range(episodes):
        state = env.reset(seed=seed + episode)
        done = False
        episode_rewards = np.zeros(env.num_firms, dtype=np.float32)
        while not done:
            actions = policy.act_all(state)
            state, rewards, done, _ = env.step(actions)
            episode_rewards += rewards[:, 0].astype(np.float32)
        scores.append(episode_rewards)
    return np.asarray(scores, dtype=np.float32)


class RuleAllPolicy:
    def __init__(self, name: str, env: BeerGameEnv, seed: int, target_inventory: int):
        self.policies = [
            build_policy(name, env.config.max_order, seed=seed + i, target_inventory=target_inventory)
            for i in range(env.num_firms)
        ]

    def act_all(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(
            [policy.act(state, firm_id) for firm_id, policy in enumerate(self.policies)],
            dtype=np.float32,
        )


class SingleAgentPolicyAll:
    def __init__(self, agent: DQNAgent, env: BeerGameEnv, seed: int):
        self.agent = agent
        self.rng = np.random.default_rng(seed)
        self.max_order = env.config.max_order
        self.num_firms = env.num_firms

    def act_all(self, state: np.ndarray) -> np.ndarray:
        actions = self.rng.integers(0, self.max_order + 1, size=self.num_firms).astype(np.float32)
        actions[self.agent.firm_id] = self.agent.act(state[self.agent.firm_id], epsilon=0.0)
        return actions


def summarize_scores(scores: np.ndarray) -> dict:
    total_scores = scores.sum(axis=1)
    summary = {
        "total_chain_mean_reward": float(np.mean(total_scores)),
        "total_chain_std_reward": float(np.std(total_scores)),
        "episodes": int(scores.shape[0]),
    }
    for firm_id in range(scores.shape[1]):
        summary[f"firm_{firm_id}_mean_reward"] = float(np.mean(scores[:, firm_id]))
        summary[f"firm_{firm_id}_std_reward"] = float(np.std(scores[:, firm_id]))
    return summary


def save_agents(agents: list[DQNAgent], model_dir: Path, seed: int):
    for agent in agents:
        path = model_dir / f"dueling_double_dqn_firm_{agent.firm_id}_seed_{seed}_tplus1.pt"
        agent.save(path)


def load_single_agent_ddqn(env: BeerGameEnv, cfg: dict, firm_id: int, seed: int) -> DQNAgent:
    model_dir = Path(cfg["experiment"].get("model_dir", "models/baselines"))
    model_path = model_dir / f"dueling_double_dqn_seed_{seed}_firm_{firm_id}_tplus1.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"缺少单智能体模型: {model_path}\n"
            "请先运行: python -m beergame.run_baselines --config configs/default.json"
        )
    agent = build_agent(env, cfg, MULTIAGENT_ALGORITHM, firm_id, seed)
    agent.load(model_path)
    return agent


def main():
    parser = argparse.ArgumentParser(description="Run independent multi-agent Dueling Double DQN.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env = BeerGameEnv(make_env_config(cfg))
    multi_cfg = {**cfg["dqn"], **cfg.get("multiagent", {})}
    seed = int(multi_cfg.get("seed", cfg["env"].get("seed", 42)))
    eval_episodes = int(cfg["experiment"].get("eval_episodes", 20))
    target_inventory = int(cfg["baselines"].get("base_stock_target", env.config.initial_inventory))
    model_dir = Path(multi_cfg.get("model_dir", "models/multiagent"))
    output_dir = Path(multi_cfg.get("output_dir", "results/multiagent"))
    figure_dir = Path(multi_cfg.get("figure_dir", "figures/multiagent"))
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    agents = build_multiagent_agents(env, cfg, seed)
    model_paths = [
        model_dir / f"dueling_double_dqn_firm_{firm_id}_seed_{seed}_tplus1.pt"
        for firm_id in range(env.num_firms)
    ]
    training_scores_path = output_dir / "multiagent_training_scores.npy"
    eval_points_path = output_dir / "multiagent_eval_points.npy"
    eval_scores_path = output_dir / "multiagent_eval_scores.npy"
    if args.skip_train and all(path.exists() for path in model_paths):
        for agent, path in zip(agents, model_paths):
            agent.load(path)
        if training_scores_path.exists():
            training_scores = np.load(training_scores_path)
            plot_multiagent_training(
                training_scores,
                figure_dir / "multiagent_training_rewards.png",
                window=int(cfg["dqn"].get("plot_window", 50)),
            )
        if eval_points_path.exists() and eval_scores_path.exists():
            plot_multiagent_eval_curve(
                np.load(eval_points_path),
                np.load(eval_scores_path),
                figure_dir / "multiagent_eval_curve.png",
            )
    else:
        training_scores, eval_points, eval_scores = train_multiagent(env, agents, multi_cfg, seed)
        np.save(training_scores_path, training_scores)
        np.save(eval_points_path, eval_points)
        np.save(eval_scores_path, eval_scores)
        save_agents(agents, model_dir, seed)
        plot_multiagent_training(
            training_scores,
            figure_dir / "multiagent_training_rewards.png",
            window=int(cfg["dqn"].get("plot_window", 50)),
        )
        plot_multiagent_eval_curve(
            eval_points,
            eval_scores,
            figure_dir / "multiagent_eval_curve.png",
        )

    eval_env = BeerGameEnv(make_env_config(cfg))
    random_scores = evaluate_policy_all(
        eval_env,
        RuleAllPolicy("random", eval_env, seed, target_inventory),
        eval_episodes,
        seed,
    )
    base_stock_scores = evaluate_policy_all(
        eval_env,
        RuleAllPolicy("base_stock", eval_env, seed, target_inventory),
        eval_episodes,
        seed,
    )
    single_agent = load_single_agent_ddqn(eval_env, cfg, int(cfg["experiment"].get("firm_id", 1)), seed)
    single_agent_scores = evaluate_policy_all(
        eval_env,
        SingleAgentPolicyAll(single_agent, eval_env, seed),
        eval_episodes,
        seed,
    )
    multiagent_scores = evaluate_policy_all(
        eval_env,
        MultiAgentPolicy(agents),
        eval_episodes,
        seed,
    )

    raw_scores = {
        "random_all": random_scores,
        "base_stock_all": base_stock_scores,
        "single_agent_ddqn": single_agent_scores,
        "multiagent_ddqn": multiagent_scores,
    }
    summary = {name: summarize_scores(scores) for name, scores in raw_scores.items()}
    summary["multiagent_ddqn"]["model_paths"] = [str(path) for path in model_paths]

    with (output_dir / "multiagent_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    plot_multiagent_comparison(summary, figure_dir / "multiagent_comparison.png")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
