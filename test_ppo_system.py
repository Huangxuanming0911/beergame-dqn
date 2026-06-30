import numpy as np
import torch
from beergame.config import load_config, make_env_config
from beergame.env import BeerGameEnv
from beergame.ppo import PPOAgent
from beergame.experiments import train_ppo, train_ppo_best, evaluate_policy


class PPOPolicy:
    def __init__(self, agent):
        self.agent = agent

    def reset(self):
        self.agent.reset_history()

    def act(self, state, firm_id):
        return self.agent.eval_act(state[firm_id])


def make_agent(history_len=1):
    cfg = load_config("configs/default.json")
    env = BeerGameEnv(make_env_config(cfg))
    return PPOAgent(
        state_dim=3 * history_len,
        action_dim=env.config.max_order + 1,
        firm_id=1,
        hidden_size=256,
        rollout_episodes=4,
        update_epochs=20,
        batch_size=256,
        lr=1e-4,
        entropy_coef=0.05,
        use_reward_norm=False,
        use_state_norm=False,
        state_history_len=history_len,
    )


def run(name, train_fn, cfg_extra, seed=42):
    cfg = load_config("configs/default.json")
    env = BeerGameEnv(make_env_config(cfg))
    agent = make_agent(history_len=1)
    train_cfg = {
        "seed": seed,
        "episodes": 1000,
        "rollout_episodes": 4,
        "reward_scale": 0.001,
        "log_every": 200,
        **cfg_extra,
    }
    train_fn(env, agent, train_cfg)
    eval_result = evaluate_policy(env, PPOPolicy(agent), agent.firm_id, episodes=20, seed=seed)
    eval_mean = float(np.mean(eval_result["scores"]))
    print(f"[{name}] seed={seed} eval_mean={eval_mean:.2f}")
    return eval_mean


if __name__ == "__main__":
    run("baseline", train_ppo, {})
    run("system_reward", train_ppo, {"use_system_reward": True})
    run("best_baseline", train_ppo_best, {"eval_every": 50, "eval_episodes": 20})
    run("best_system", train_ppo_best, {"use_system_reward": True, "eval_every": 50, "eval_episodes": 20})
