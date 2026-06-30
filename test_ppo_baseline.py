import numpy as np
import torch
from beergame.config import load_config, make_env_config
from beergame.env import BeerGameEnv
from beergame.ppo import PPOAgent
from beergame.experiments import train_ppo, evaluate_policy


class PPOPolicy:
    def __init__(self, agent):
        self.agent = agent

    def act(self, state, firm_id):
        with torch.no_grad():
            state_t = torch.FloatTensor(state[firm_id]).unsqueeze(0).to(self.agent.device)
            logits, _ = self.agent.net(state_t)
            return int(logits.argmax(dim=-1).item())


def run(seed=42):
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
        use_reward_norm=False,
        use_state_norm=False,
    )
    train_cfg = {
        "seed": seed,
        "episodes": 1000,
        "rollout_episodes": 4,
        "reward_scale": 0.001,
        "log_every": 200,
    }
    train_ppo(env, agent, train_cfg)
    eval_result = evaluate_policy(env, PPOPolicy(agent), agent.firm_id, episodes=20, seed=seed)
    eval_mean = float(np.mean(eval_result["scores"]))
    print(f"seed={seed} eval_mean={eval_mean:.2f}")


if __name__ == "__main__":
    run(seed=42)
