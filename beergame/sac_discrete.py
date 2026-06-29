"""Discrete Soft Actor-Critic (SAC) for the beer game environment.

SAC maximizes a trade-off between expected return and policy entropy.  The
policy is updated using the policy-gradient theorem with entropy regularization,
so it stays in the policy-gradient family while leveraging a replay buffer and
twin Q-networks for stability.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


class QNetwork(nn.Module):
    """Q-network for discrete actions."""

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PolicyNetwork(nn.Module):
    """Stochastic policy network for discrete actions."""

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.logits = nn.Linear(hidden_size, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits(self.net(x))

    def sample(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def get_action_and_log_probs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        probs = dist.probs
        # Entropy-regularized policy objective: sum_a π(a|s) * (Q(s,a) - α log π(a|s))
        # We return action, log_prob, and probs for the policy update.
        return action, log_prob, probs


class ReplayBuffer:
    """Simple replay buffer for off-policy learning."""

    def __init__(self, state_dim: int, action_dim: int, capacity: int = 100000):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        idx = self.ptr
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        indices = np.random.choice(self.size, batch_size, replace=False)
        return (
            torch.FloatTensor(self.states[indices]),
            torch.LongTensor(self.actions[indices]),
            torch.FloatTensor(self.rewards[indices]),
            torch.FloatTensor(self.next_states[indices]),
            torch.FloatTensor(self.dones[indices]),
        )


class DiscreteSACAgent:
    """Discrete SAC agent compatible with the baseline runner."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        firm_id: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        hidden_size: int = 64,
        buffer_size: int = 100000,
        batch_size: int = 64,
        update_every: int = 1,
        reward_scale: float = 1.0,
        device: str | None = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.firm_id = firm_id
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.update_every = update_every
        self.reward_scale = reward_scale
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.q1 = QNetwork(state_dim, action_dim, hidden_size).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim, hidden_size).to(self.device)
        self.q1_target = QNetwork(state_dim, action_dim, hidden_size).to(self.device)
        self.q2_target = QNetwork(state_dim, action_dim, hidden_size).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_size).to(self.device)

        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.buffer = ReplayBuffer(state_dim, action_dim, buffer_size)
        self.step_count = 0

    def act(self, state: np.ndarray, epsilon: float | None = None) -> int:
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, _ = self.policy.sample(state_t)
        return int(action.item())

    def step(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward * self.reward_scale, next_state, done)
        self.step_count += 1

        if self.step_count % self.update_every == 0 and self.buffer.size >= self.batch_size:
            return self.update()
        return {}

    def update(self) -> dict[str, float]:
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Q target
        with torch.no_grad():
            next_logits = self.policy.forward(next_states)
            next_dist = Categorical(logits=next_logits)
            next_probs = next_dist.probs
            next_log_probs = torch.log(next_probs + 1e-8)
            next_q1 = self.q1_target(next_states)
            next_q2 = self.q2_target(next_states)
            next_q = torch.min(next_q1, next_q2)
            next_v = (next_probs * (next_q - self.alpha * next_log_probs)).sum(dim=-1)
            target_q = rewards + self.gamma * (1.0 - dones) * next_v

        # Update Q networks
        q1_values = self.q1(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        q2_values = self.q2(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        q1_loss = F.mse_loss(q1_values, target_q)
        q2_loss = F.mse_loss(q2_values, target_q)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Update policy
        logits = self.policy.forward(states)
        dist = Categorical(logits=logits)
        probs = dist.probs
        log_probs = torch.log(probs + 1e-8)
        with torch.no_grad():
            q1 = self.q1(states)
            q2 = self.q2(states)
            q = torch.min(q1, q2)
        policy_loss = (probs * (self.alpha * log_probs - q)).sum(dim=-1).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Soft update targets
        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "policy_loss": policy_loss.item(),
        }

    def _soft_update(self, source: nn.Module, target: nn.Module):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "policy": self.policy.state_dict(),
                "q1_optimizer": self.q1_optimizer.state_dict(),
                "q2_optimizer": self.q2_optimizer.state_dict(),
                "policy_optimizer": self.policy_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.q1.load_state_dict(checkpoint["q1"])
        self.q2.load_state_dict(checkpoint["q2"])
        self.policy.load_state_dict(checkpoint["policy"])
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.q1_optimizer.load_state_dict(checkpoint["q1_optimizer"])
        self.q2_optimizer.load_state_dict(checkpoint["q2_optimizer"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
