from __future__ import annotations

import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """小型MLP：把单个企业的3维观测映射为各动作的Q值。"""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class DuelingQNetwork(nn.Module):
    """Dueling结构：分别估计状态价值V(s)和动作优势A(s,a)。"""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.value = nn.Linear(hidden_size, 1)
        self.advantage = nn.Linear(hidden_size, action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        feature = self.feature(state)
        value = self.value(feature)
        advantage = self.advantage(feature)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


def build_q_network(network_type: str, state_size: int, action_size: int, hidden_size: int) -> nn.Module:
    if network_type == "standard":
        return QNetwork(state_size, action_size, hidden_size)
    if network_type == "dueling":
        return DuelingQNetwork(state_size, action_size, hidden_size)
    raise ValueError(f"未知网络类型: {network_type}")


class ReplayBuffer:
    """经验回放池：打散样本时间相关性，稳定DQN更新。"""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        firm_id: int,
        hidden_size: int = 64,
        buffer_size: int = 10000,
        batch_size: int = 64,
        gamma: float = 0.99,
        learning_rate: float = 1e-3,
        tau: float = 1e-3,
        update_every: int = 4,
        network_type: str = "standard",
        double_dqn: bool = False,
        seed: int | None = 42,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.firm_id = firm_id
        self.network_type = network_type
        self.double_dqn = double_dqn
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.t_step = 0
        random.seed(seed)
        torch.manual_seed(seed or 0)

        self.q_network = build_q_network(network_type, state_size, action_size, hidden_size)
        self.target_network = build_q_network(network_type, state_size, action_size, hidden_size)
        # 目标网络提供变化较慢的bootstrap目标，降低训练震荡。
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_size)

    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        # 训练时使用epsilon-greedy探索；评估时epsilon=0，选择贪心动作。
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        state_tensor = torch.from_numpy(state.reshape(1, -1)).float()
        self.q_network.eval()
        with torch.no_grad():
            action_values = self.q_network(state_tensor)
        self.q_network.train()
        return int(torch.argmax(action_values, dim=1).item())

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) >= self.batch_size:
            self.learn(self.memory.sample(self.batch_size))

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.from_numpy(np.vstack(states)).float()
        actions = torch.from_numpy(np.vstack(actions)).long()
        rewards = torch.from_numpy(np.vstack(rewards)).float()
        next_states = torch.from_numpy(np.vstack(next_states)).float()
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float()

        if self.double_dqn:
            # Double DQN：在线网络选动作，目标网络估值，缓解Q值高估。
            next_actions = self.q_network(next_states).detach().argmax(1).unsqueeze(1)
            q_targets_next = self.target_network(next_states).detach().gather(1, next_actions)
        else:
            # 普通DQN：直接使用目标网络中下一状态的最大Q值。
            q_targets_next = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + self.gamma * q_targets_next * (1 - dones)
        q_expected = self.q_network(states).gather(1, actions)
        loss = nn.MSELoss()(q_expected, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update()

    def soft_update(self):
        # Polyak平均让目标网络缓慢跟随当前网络。
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q_network_state_dict": self.q_network.state_dict(),
                "target_network_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "firm_id": self.firm_id,
                "state_size": self.state_size,
                "action_size": self.action_size,
                "network_type": self.network_type,
                "double_dqn": self.double_dqn,
            },
            path,
        )

    def load(self, path: str | Path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
