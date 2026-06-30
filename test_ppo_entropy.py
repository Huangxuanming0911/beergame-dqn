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


def run(ent):
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
        entropy_coef=ent,
        target_kl=0.015,
        use_reward_norm=False,
        use_state_norm=False,
    )
    train_ppo(env, agent, {"seed": 42, "episodes": 1000, "rollout_episodes": 4, "reward_scale": 0.001, "log_every": 200})
    res = evaluate_policy(env, PPOPolicy(agent), 1, 20, seed=42)
    print(f"entropy={ent} eval_mean={np.mean(res['scores']):.2f}")


for ent in [0.01, 0.02, 0.1]:
    run(ent)
