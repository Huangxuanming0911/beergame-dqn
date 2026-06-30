import sys
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


def main():
    cfg = load_config("configs/default.json")
    env = BeerGameEnv(make_env_config(cfg))

    variant = sys.argv[1]
    seed = int(sys.argv[2])

    kwargs = {
        "state_dim": 3,
        "action_dim": env.config.max_order + 1,
        "firm_id": 1,
        "hidden_size": 256,
        "rollout_episodes": 4,
        "update_epochs": 20,
        "batch_size": 256,
    }
    episodes = 1000

    if variant == "default":
        pass
    elif variant == "lr3e4":
        kwargs["lr"] = 3e-4
    elif variant == "ent_high":
        kwargs["entropy_coef"] = 0.1
    elif variant == "lambda99":
        kwargs["gae_lambda"] = 0.99
    elif variant == "clip01":
        kwargs["clip_epsilon"] = 0.1
    elif variant == "epochs30":
        kwargs["update_epochs"] = 30
    elif variant == "hidden512":
        kwargs["hidden_size"] = 512
    else:
        raise ValueError(variant)

    agent = PPOAgent(**kwargs)
    train_ppo(
        env,
        agent,
        {
            "seed": seed,
            "episodes": episodes,
            "rollout_episodes": 4,
            "reward_scale": 0.001,
            "log_every": 200,
        },
    )
    eval_result = evaluate_policy(env, PPOPolicy(agent), agent.firm_id, episodes=20, seed=seed)
    eval_mean = float(np.mean(eval_result["scores"]))
    print(f"{variant} seed={seed} eval_mean={eval_mean:.2f}")


if __name__ == "__main__":
    main()
