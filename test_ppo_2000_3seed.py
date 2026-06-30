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


def run(seed):
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
        entropy_coef=0.05,
        target_kl=0.015,
        use_reward_norm=False,
        use_state_norm=False,
    )
    train_ppo(env, agent, {"seed": seed, "episodes": 2000, "rollout_episodes": 4, "reward_scale": 0.001, "log_every": 300})
    eval_result = evaluate_policy(env, PPOPolicy(agent), agent.firm_id, episodes=20, seed=seed)
    eval_mean = float(np.mean(eval_result["scores"]))
    print(f"seed={seed} eval_mean={eval_mean:.2f}")
    return eval_mean


if __name__ == "__main__":
    means = []
    for s in [42, 123, 456]:
        means.append(run(s))
    print("mean", np.mean(means), "std", np.std(means))
