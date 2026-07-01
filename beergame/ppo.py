from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """共享主体的 Actor-Critic 网络，用于离散动作空间。"""

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(hidden_size, action_dim)
        self.critic_head = nn.Linear(hidden_size, 1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        return self.actor_head(h), self.critic_head(h)

    def act(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value.squeeze(-1)

    def evaluate(self, x: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        return dist.log_prob(action), value.squeeze(-1), dist.entropy()


class RunningMeanStd:
    """在线维护运行均值与方差，用于状态与奖励归一化。"""

    def __init__(self, epsilon: float = 1e-4, shape: tuple[int, ...] = ()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        tot_count = self.count + batch_count
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_count / tot_count
        m2 = (
            self.var * self.count
            + batch_var * batch_count
            + delta**2 * self.count * batch_count / tot_count
        )
        self.mean = new_mean
        self.var = m2 / tot_count
        self.count = tot_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


class PPOAgent:
    """离散动作 PPO 智能体，支持 GAE、值函数裁剪、状态/奖励归一化、KL 早停与熵退火。"""

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
        target_kl: float | None = 0.015,
        rollout_episodes: int = 4,
        use_reward_norm: bool = True,
        use_state_norm: bool = True,
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
        self.target_kl = target_kl
        self.rollout_episodes = rollout_episodes
        self.use_reward_norm = use_reward_norm
        self.use_state_norm = use_state_norm
        self.use_value_clip = use_value_clip
        self.use_lr_decay = use_lr_decay
        self.use_entropy_decay = use_entropy_decay
        self.total_updates = total_updates
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.update_count = 0

        self.net = ActorCritic(state_dim, action_dim, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.initial_lr = lr

        self.state_rms = RunningMeanStd(shape=(state_dim,))
        self.reward_rms = RunningMeanStd(shape=())

        self.states: list[np.ndarray] = []
        self.actions: list[int] = []
        self.log_probs: list[float] = []
        self.values: list[float] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.episodes_since_update = 0

    def act(self, state: np.ndarray) -> int:
        norm_state = self._normalize_state(state)
        with torch.no_grad():
            state_t = torch.FloatTensor(norm_state).unsqueeze(0).to(self.device)
            action, log_prob, value = self.net.act(state_t)
        self.states.append(norm_state.copy())
        self.actions.append(int(action.item()))
        self.log_probs.append(float(log_prob.item()))
        self.values.append(float(value.item()))
        return self.actions[-1]

    def eval_act(self, state: np.ndarray) -> int:
        norm_state = self._normalize_state(state)
        with torch.no_grad():
            state_t = torch.FloatTensor(norm_state).unsqueeze(0).to(self.device)
            logits = self.net.forward(state_t)[0]
            action = logits.argmax(dim=-1)
        return int(action.item())

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        state = state.astype(np.float32, copy=False).reshape(self.state_dim)
        if self.use_state_norm:
            self.state_rms.update(state.reshape(1, -1))
            return self.state_rms.normalize(state)
        return state

    def reset_history(self) -> None:
        pass

    def store_transition(self, reward: float, done: bool) -> None:
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    def should_update(self, done: bool) -> bool:
        if done:
            self.episodes_since_update += 1
        return self.episodes_since_update >= self.rollout_episodes

    def update(
        self,
        next_state: np.ndarray | None = None,
        next_critic_state: np.ndarray | None = None,
        next_values: np.ndarray | None = None,
    ) -> dict[str, float]:
        if len(self.states) == 0:
            return {}

        self._update_lr()
        self._update_entropy()

        rewards = self._normalize_rewards()
        next_value = self._estimate_next_value(next_state)
        advantages, returns = self._compute_gae(next_value, rewards)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = torch.FloatTensor(np.asarray(self.states)).to(self.device)
        actions = torch.LongTensor(np.asarray(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.asarray(self.log_probs)).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        old_values = torch.FloatTensor(np.asarray(self.values)).to(self.device)

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_batches = 0

        for _ in range(self.update_epochs):
            indices = np.arange(len(self.states))
            np.random.shuffle(indices)
            for start in range(0, len(self.states), self.batch_size):
                batch_idx = indices[start : start + self.batch_size]

                log_probs, values, entropy = self.net.evaluate(
                    states[batch_idx], actions[batch_idx]
                )
                ratio = torch.exp(log_probs - old_log_probs[batch_idx])
                surr1 = ratio * advantages[batch_idx]
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * advantages[batch_idx]
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                if self.use_value_clip:
                    value_clipped = old_values[batch_idx] + torch.clamp(
                        values - old_values[batch_idx],
                        -self.clip_epsilon,
                        self.clip_epsilon,
                    )
                    value_loss = torch.max(
                        nn.functional.mse_loss(values, returns[batch_idx]),
                        nn.functional.mse_loss(value_clipped, returns[batch_idx]),
                    ).mean()
                else:
                    value_loss = nn.functional.mse_loss(values, returns[batch_idx])

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy.mean()
                )

                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += float(loss.item())
                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy.mean().item())
                num_batches += 1

            if self.target_kl is not None:
                with torch.no_grad():
                    new_log_probs, _, _ = self.net.evaluate(states, actions)
                    kl = float((old_log_probs - new_log_probs).mean().item())
                if kl > self.target_kl:
                    break

        self._clear_buffer()
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

    def _normalize_rewards(self) -> np.ndarray:
        rewards = np.asarray(self.rewards, dtype=np.float32)
        if self.use_reward_norm:
            self.reward_rms.update(rewards.reshape(-1, 1))
            rewards = self.reward_rms.normalize(rewards.reshape(-1, 1)).flatten()
        return rewards

    def _estimate_next_value(self, next_state: np.ndarray | None) -> float:
        if next_state is None:
            return 0.0
        state = next_state.astype(np.float32, copy=False).reshape(self.state_dim)
        if self.use_state_norm:
            self.state_rms.update(state.reshape(1, -1))
            state = self.state_rms.normalize(state)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            _, value = self.net(state_t)
        return float(value.item())

    def _compute_gae(self, next_value: float, rewards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(self.values, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        advantages = np.zeros_like(rewards)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            next_v = next_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_v * (1.0 - dones[t]) - values[t]
            last_gae = (
                delta
                + self.gamma * self.gae_lambda * (1.0 - dones[t]) * last_gae
            )
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
        self.entropy_coef = self.entropy_coef_start * (0.1 + 0.9 * (1.0 - progress))

    def _clear_buffer(self) -> None:
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
                "state_rms_mean": self.state_rms.mean.copy(),
                "state_rms_var": self.state_rms.var.copy(),
                "state_rms_count": float(self.state_rms.count),
                "reward_rms_mean": float(self.reward_rms.mean.reshape(-1)[0]),
                "reward_rms_var": float(self.reward_rms.var.reshape(-1)[0]),
                "reward_rms_count": float(self.reward_rms.count),
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(checkpoint["net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "state_rms_mean" in checkpoint:
            self.state_rms.mean = np.asarray(checkpoint["state_rms_mean"])
            self.state_rms.var = np.asarray(checkpoint["state_rms_var"])
            self.state_rms.count = float(checkpoint["state_rms_count"])
        self.reward_rms.mean = np.asarray(checkpoint.get("reward_rms_mean", 0.0))
        self.reward_rms.var = np.asarray(checkpoint.get("reward_rms_var", 1.0))
        self.reward_rms.count = float(checkpoint.get("reward_rms_count", 1e-4))
