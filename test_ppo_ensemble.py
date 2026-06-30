import numpy as np
import torch
from beergame.config import load_config, make_env_config
from beergame.env import BeerGameEnv
from beergame.ppo import PPOAgent
from beergame.experiments import train_ppo, evaluate_policy


class EnsemblePPOPolicy:
    def __init__(self, agents):
        self.agents = agents

    def reset(self):
        for a in self.agents:
            a.reset_history()

    def act(self, state, firm_id):
        with torch.no_grad():
            logits_sum = None
            for agent in self.agents:
                state_t = torch.FloatTensor(state[firm_id]).unsqueeze(0).to(agent.device)
                logits, _ = agent.net(state_t)
                logits_sum = logits if logits_sum is None else logits_sum + logits
            action = logits_sum.argmax(dim=-1)
        return int(action.item())


def make_agent(seed_offset=0):
    cfg = load_config("configs/default.json")
    env = BeerGameEnv(make_env_config(cfg))
    return PPOAgent(
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


def train_agent(seed):
    cfg = load_config("configs/default.json")
    env = BeerGameEnv(make_env_config(cfg))
    agent = make_agent()
    train_ppo(env, agent, {"seed": seed, "episodes": 1000, "rollout_episodes": 4,
                           "reward_scale": 0.001, "log_every": 200})
    return agent


if __name__ == "__main__":
    seed = 42
    agents = [train_agent(seed + i * 10000) for i in range(3)]
    cfg = load_config("configs/default.json")
    env = BeerGameEnv(make_env_config(cfg))
    # individual scores
    class P:
        def __init__(self, a):
            self.a = a
        def reset(self):
            self.a.reset_history()
        def act(self, state, fid):
            return self.a.eval_act(state[fid])
    for i, a in enumerate(agents):
        res = evaluate_policy(env, P(a), 1, 20, seed=seed)
        print(f"agent {i} eval_mean={np.mean(res['scores']):.2f}")
    ens = evaluate_policy(env, EnsemblePPOPolicy(agents), 1, 20, seed=seed)
    print(f"ensemble eval_mean={np.mean(ens['scores']):.2f}")
