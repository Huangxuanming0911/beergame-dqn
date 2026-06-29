"""Discrete-action PPO agent for the beer game environment.

Improvements over a naive PPO:
- running reward normalization
- multi-episode rollouts (larger update batches)
- clipped value loss (PPO-style)
- linear learning-rate and entropy decay
- gradient clipping
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """Actor-critic network for discrete actions.

    Supports both shared-body and separate-body architectures.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 64,
        separate_actor_critic: bool = False,
    ):
        super().__init__()
        self.separate_actor_critic = separate_actor_critic
        if separate_actor_critic:
            self.actor_body = nn.Sequential(
                nn.Linear(state_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
            )
            self.critic_body = nn.Sequential(
                nn.Linear(state_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
            )
        else:
            self.shared = nn.Sequential(
                nn.Linear(state_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
            )
        self.actor_head = nn.Linear(hidden_size, action_dim)
        self.critic_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.separate_actor_critic:
            h_actor = self.actor_body(x)
            h_critic = self.critic_body(x)
            logits = self.actor_head(h_actor)
            value = self.critic_head(h_critic)
        else:
            h = self.shared(x)
            logits = self.actor_head(h)
            value = self.critic_head(h)
        return logits, value

    def act(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value.squeeze(-1)

    def evaluate(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, value.squeeze(-1), entropy


class RunningMeanStd:
    """Running mean and standard deviation for online normalization."""

    def __init__(self, epsilon: float = 1e-4, shape: tuple[int, ...] = ()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = m2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


class PPOAgent:
    """PPO agent that can be plugged into the existing baseline runner."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        firm_id: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.05,
        hidden_size: int = 64,
        update_epochs: int = 10,
        batch_size: int = 256,
        max_grad_norm: float = 0.5,
        separate_actor_critic: bool = False,
        rollout_episodes: int = 4,
        use_reward_norm: bool = True,
        use_value_clip: bool = True,
        use_lr_decay: bool = True,
        use_entropy_decay: bool = True,
        total_updates: int | None = None,
        device: str | None = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.firm_id = firm_id
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef_start = entropy_coef
        self.entropy_coef = entropy_coef
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.rollout_episodes = rollout_episodes
        self.use_reward_norm = use_reward_norm
        self.use_value_clip = use_value_clip
        self.use_lr_decay = use_lr_decay
        self.use_entropy_decay = use_entropy_decay
        self.total_updates = total_updates
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.update_count = 0

        self.net = ActorCritic(
            state_dim, action_dim, hidden_size, separate_actor_critic=separate_actor_critic
        ).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.initial_lr = lr

        # Running reward normalizer
        self.reward_rms = RunningMeanStd(shape=())

        # Rollout buffer
        self.states: list[np.ndarray] = []
        self.actions: list[int] = []
        self.log_probs: list[float] = []
        self.values: list[float] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.episodes_since_update = 0

    def act(self, state: np.ndarray, epsilon: float | None = None) -> int:
        """Select an action for the current state (rollout)."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, value = self.net.act(state_t)
        action = int(action.item())
        log_prob = float(log_prob.item())
        value = float(value.item())

        self.states.append(state.copy())
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        return action

    def store_transition(self, reward: float, done: bool) -> None:
        """Store reward and done signal after the environment step."""
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    def _normalize_rewards(self) -> np.ndarray:
        rewards = np.asarray(self.rewards, dtype=np.float32)
        if self.use_reward_norm:
            self.reward_rms.update(rewards.reshape(-1, 1))
            rewards = self.reward_rms.normalize(rewards.reshape(-1, 1)).flatten()
        return rewards

    def _compute_gae(self, next_value: float, rewards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute returns and advantages using GAE."""
        values = np.asarray(self.values, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        advantages = np.zeros_like(rewards)
        last_gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_v = next_value
            else:
                next_v = values[t + 1]
            delta = rewards[t] + self.gamma * next_v * (1.0 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def _update_lr(self) -> None:
        if not self.use_lr_decay or self.total_updates is None or self.total_updates <= 1:
            return
        progress = min(1.0, self.update_count / (self.total_updates - 1))
        lr = self.initial_lr * (1.0 - progress)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _update_entropy(self) -> None:
        if not self.use_entropy_decay or self.total_updates is None or self.total_updates <= 1:
            return
        progress = min(1.0, self.update_count / (self.total_updates - 1))
        # Decay to 10% of initial entropy coefficient
        self.entropy_coef = self.entropy_coef_start * (0.1 + 0.9 * (1.0 - progress))

    def should_update(self, done: bool) -> bool:
        """Return True when enough episodes have been collected."""
        if done:
            self.episodes_since_update += 1
        return self.episodes_since_update >= self.rollout_episodes

    def update(self, next_state: np.ndarray | None = None) -> dict[str, float]:
        """Perform a PPO update using the collected rollout buffer."""
        if len(self.states) == 0:
            return {}

        self._update_lr()
        self._update_entropy()

        with torch.no_grad():
            if next_state is not None:
                next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                _, next_value = self.net.forward(next_state_t)
                next_value = float(next_value.item())
            else:
                next_value = 0.0

        rewards = self._normalize_rewards()
        advantages, returns = self._compute_gae(next_value, rewards)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = torch.FloatTensor(np.asarray(self.states)).to(self.device)
        actions = torch.LongTensor(np.asarray(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.asarray(self.log_probs)).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        old_values = torch.FloatTensor(np.asarray(self.values)).to(self.device)

        dataset_size = len(self.states)
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_batches = 0

        for _ in range(self.update_epochs):
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_old_values = old_values[batch_idx]

                log_probs, values, entropy = self.net.evaluate(batch_states, batch_actions)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                if self.use_value_clip:
                    value_pred_clipped = batch_old_values + torch.clamp(
                        values - batch_old_values, -self.clip_epsilon, self.clip_epsilon
                    )
                    value_loss_1 = nn.functional.mse_loss(values, batch_returns)
                    value_loss_2 = nn.functional.mse_loss(value_pred_clipped, batch_returns)
                    value_loss = torch.max(value_loss_1, value_loss_2).mean()
                else:
                    value_loss = nn.functional.mse_loss(values, batch_returns)

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_batches += 1

        # Clear buffer and counters
        self.clear_buffer()
        self.update_count += 1

        if num_batches == 0:
            return {}
        return {
            "loss": total_loss / num_batches,
            "policy_loss": total_policy_loss / num_batches,
            "value_loss": total_value_loss / num_batches,
            "entropy": total_entropy / num_batches,
            "lr": self.optimizer.param_groups[0]["lr"],
            "ent_coef": self.entropy_coef,
        }

    def clear_buffer(self) -> None:
        """Clear the rollout buffer."""
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
        self.episodes_since_update = 0

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "net_state_dict": self.net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "firm_id": self.firm_id,
                "reward_rms_mean": float(self.reward_rms.mean),
                "reward_rms_var": float(self.reward_rms.var),
                "reward_rms_count": float(self.reward_rms.count),
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.net.load_state_dict(checkpoint["net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.reward_rms.mean = checkpoint.get("reward_rms_mean", 0.0)
        self.reward_rms.var = checkpoint.get("reward_rms_var", 1.0)
        self.reward_rms.count = checkpoint.get("reward_rms_count", 1e-4)
