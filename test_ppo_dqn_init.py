import numpy as np
import torch
import torch.nn as nn
from beergame.config import load_config, make_env_config
from beergame.env import BeerGameEnv
from beergame.dqn import DQNAgent
from beergame.ppo import PPOAgent
from beergame.experiments import train_ppo, evaluate_policy, make_background_actions


class PPOPolicy:
    def __init__(self, agent):
        self.agent = agent

    def reset(self):
        self.agent.reset_history()

    def act(self, state, firm_id):
        return self.agent.eval_act(state[firm_id])


def load_dqn(seed):
    agent = DQNAgent(
        state_size=3,
        action_size=21,
        firm_id=1,
        hidden_size=64,
        network_type="dueling",
        double_dqn=True,
        seed=seed,
    )
    agent.load(f"models/baselines/dueling_double_dqn_seed_{seed}_firm_1_tplus1.pt")
    return agent


def collect_demos(env, dqn, seed, episodes=50):
    rng = np.random.default_rng(seed)
    states, actions = [], []
    for ep in range(episodes):
        state = env.reset(seed=seed + ep)
        done = False
        while not done:
            actions_bg = make_background_actions(env, state, dqn.firm_id, rng, "random")
            actions_bg[dqn.firm_id] = float(dqn.act(state[dqn.firm_id], epsilon=0.0))
            states.append(state[dqn.firm_id].copy())
            actions.append(int(actions_bg[dqn.firm_id]))
            state, _, done, _ = env.step(actions_bg)
    return np.asarray(states, dtype=np.float32), np.asarray(actions, dtype=np.int64)


def pretrain_ppo_actor(ppo, states, actions, epochs=50, batch_size=256):
    optimizer = torch.optim.Adam(ppo.net.parameters(), lr=1e-4)
    dataset_size = len(states)
    states_t = torch.FloatTensor(states).to(ppo.device)
    actions_t = torch.LongTensor(actions).to(ppo.device)
    for _ in range(epochs):
        indices = np.arange(dataset_size)
        np.random.shuffle(indices)
        for start in range(0, dataset_size, batch_size):
            idx = indices[start:start + batch_size]
            logits, _ = ppo.net.forward(states_t[idx], None)
            loss = nn.functional.cross_entropy(logits, actions_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def run(seed):
    cfg = load_config("configs/default.json")
    env = BeerGameEnv(make_env_config(cfg))
    dqn = load_dqn(seed)
    states, actions = collect_demos(env, dqn, seed, episodes=50)
    ppo = PPOAgent(
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
    pretrain_ppo_actor(ppo, states, actions, epochs=50, batch_size=256)
    train_ppo(env, ppo, {"seed": seed, "episodes": 1000, "rollout_episodes": 4,
                         "reward_scale": 0.001, "log_every": 200})
    eval_result = evaluate_policy(env, PPOPolicy(ppo), ppo.firm_id, episodes=20, seed=seed)
    eval_mean = float(np.mean(eval_result["scores"]))
    print(f"seed={seed} eval_mean={eval_mean:.2f}")
    return eval_mean


if __name__ == "__main__":
    for s in [42, 123, 456]:
        run(s)
