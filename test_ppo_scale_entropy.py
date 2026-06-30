import numpy as np
import torch
from beergame.config import load_config, make_env_config
from beergame.env import BeerGameEnv
from beergame.ppo import PPOAgent
from beergame.experiments import train_ppo, evaluate_policy


class PPOPolicy:
    def __init__(self, agent):
        self.agent = agent

    def reset(self):
        self.agent.reset_history()

    def act(self, state, firm_id):
        return self.agent.eval_act(state[firm_id])


def run(seed=42, reward_scale=0.001, entropy=0.05):
    cfg = load_config("configs/default.json")
    env = BeerGameEnv(make_env_config(cfg))
    agent = PPOAgent(
        state_dim=3,
        action_dim=env.config.max_order + 1,
        firm_id=1,
        hidden_size=256,
        rollout_episodes=4,
        update_epochs=20,
        batch_size=256,
        lr=1e-4,
        entropy_coef=entropy,
        target_kl=0.015,
        use_reward_norm=False,
        use_state_norm=False,
    )
    train_cfg = {
        "seed": seed,
        "episodes": 1000,
        "rollout_episodes": 4,
        "reward_scale": reward_scale,
        "log_every": 200,
    }
    train_ppo(env, agent, train_cfg)
    eval_result = evaluate_policy(env, PPOPolicy(agent), agent.firm_id, episodes=20, seed=seed)
    eval_mean = float(np.mean(eval_result["scores"]))
    print(f"rs={reward_scale} ent={entropy} seed={seed} eval_mean={eval_mean:.2f}")
    return eval_mean


if __name__ == "__main__":
    for rs in [0.0001, 0.001, 0.01, 0.1, 1.0]:
        run(reward_scale=rs)
    for ent in [0.0, 0.01, 0.02, 0.05, 0.1]:
        run(entropy=ent)
