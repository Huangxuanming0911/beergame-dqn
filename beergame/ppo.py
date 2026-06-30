"""Discrete-action PPO agent for the beer game environment.

Improvements over a naive PPO:
- running reward normalization
- multi-episode rollouts (larger update batches)
- clipped value loss (PPO-style)
- linear learning-rate and entropy decay
- gradient clipping
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


def _make_mlp_layers(
    in_dim: int,
    hidden_size: int,
    activation: str = "relu",
    use_layer_norm: bool = False,
) -> list[nn.Module]:
    """Build two hidden-layer MLP with optional LayerNorm."""
    act_cls: type[nn.Module]
    if activation.lower() == "elu":
        act_cls = nn.ELU
    elif activation.lower() == "tanh":
        act_cls = nn.Tanh
    else:
        act_cls = nn.ReLU

    layers: list[nn.Module] = [
        nn.Linear(in_dim, hidden_size),
    ]
    if use_layer_norm:
        layers.append(nn.LayerNorm(hidden_size))
    layers.append(act_cls())
    layers.append(nn.Linear(hidden_size, hidden_size))
    if use_layer_norm:
        layers.append(nn.LayerNorm(hidden_size))
    layers.append(act_cls())
    return layers


def orthogonal_init(m: nn.Module, gain: float = 1.0) -> None:
    """Orthogonal weight initialization for Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain=gain)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


class ActorCritic(nn.Module):
    """Actor-critic network for discrete actions.

    Supports both shared-body and separate-body architectures, and allows the
    critic to observe a different (e.g., centralized) state than the actor.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 64,
        separate_actor_critic: bool = False,
        critic_state_dim: int | None = None,
        activation: str = "relu",
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.separate_actor_critic = separate_actor_critic
        critic_state_dim = critic_state_dim if critic_state_dim is not None else state_dim
        if separate_actor_critic:
            self.actor_body = nn.Sequential(
                *_make_mlp_layers(state_dim, hidden_size, activation, use_layer_norm)
            )
            self.critic_body = nn.Sequential(
                *_make_mlp_layers(critic_state_dim, hidden_size, activation, use_layer_norm)
            )
        else:
            if critic_state_dim != state_dim:
                raise ValueError("shared-body ActorCritic requires critic_state_dim == state_dim")
            self.shared = nn.Sequential(
                *_make_mlp_layers(state_dim, hidden_size, activation, use_layer_norm)
            )
        self.actor_head = nn.Linear(hidden_size, action_dim)
        self.critic_head = nn.Linear(hidden_size, 1)
        self.apply(lambda m: orthogonal_init(m, gain=nn.init.calculate_gain("relu")))

    def forward(self, x: torch.Tensor, critic_x: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if self.separate_actor_critic:
            h_actor = self.actor_body(x)
            h_critic = self.critic_body(critic_x if critic_x is not None else x)
            logits = self.actor_head(h_actor)
            value = self.critic_head(h_critic)
        else:
            h = self.shared(x)
            logits = self.actor_head(h)
            value = self.critic_head(h)
        return logits, value

    def get_action_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return only the actor logits (used for deterministic evaluation)."""
        if self.separate_actor_critic:
            return self.actor_head(self.actor_body(x))
        return self.actor_head(self.shared(x))

    def act(self, x: torch.Tensor, critic_x: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(x, critic_x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value.squeeze(-1)

    def evaluate(
        self,
        x: torch.Tensor,
        action: torch.Tensor,
        critic_x: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(x, critic_x)
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
        target_kl: float | None = 0.015,
        separate_actor_critic: bool = False,
        rollout_episodes: int = 4,
        use_reward_norm: bool = True,
        use_state_norm: bool = True,
        use_value_clip: bool = True,
        use_lr_decay: bool = True,
        use_entropy_decay: bool = True,
        state_history_len: int = 1,
        use_ema: bool = False,
        ema_tau: float = 0.005,
        centralized_critic: bool = False,
        critic_state_dim: int | None = None,
        total_updates: int | None = None,
        device: str | None = None,
        activation: str = "relu",
        use_layer_norm: bool = False,
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
        self.state_history_len = state_history_len
        self.use_ema = use_ema
        self.ema_tau = ema_tau
        self.centralized_critic = centralized_critic
        self.critic_state_dim = critic_state_dim if critic_state_dim is not None else state_dim
        self.total_updates = total_updates
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.activation = activation
        self.use_layer_norm = use_layer_norm
        self.update_count = 0

        # Centralized critic needs separate actor/critic bodies with different inputs.
        if centralized_critic:
            separate_actor_critic = True

        self.net = ActorCritic(
            state_dim,
            action_dim,
            hidden_size,
            separate_actor_critic=separate_actor_critic,
            critic_state_dim=self.critic_state_dim,
            activation=activation,
            use_layer_norm=use_layer_norm,
        ).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.initial_lr = lr

        # Exponential moving average of the network for stable evaluation.
        if use_ema:
            self.ema_net = ActorCritic(
                state_dim,
                action_dim,
                hidden_size,
                separate_actor_critic=separate_actor_critic,
                critic_state_dim=self.critic_state_dim,
                activation=activation,
                use_layer_norm=use_layer_norm,
            ).to(self.device)
            self.ema_net.load_state_dict(self.net.state_dict())
            for p in self.ema_net.parameters():
                p.requires_grad = False
        else:
            self.ema_net = None

        # Running normalizers
        self.state_rms = RunningMeanStd(shape=(state_dim,))
        if centralized_critic:
            self.critic_state_rms = RunningMeanStd(shape=(self.critic_state_dim,))
        self.reward_rms = RunningMeanStd(shape=())

        # Rollout buffer
        self.states: list[np.ndarray] = []
        self.critic_states: list[np.ndarray] = []
        self.actions: list[int] = []
        self.log_probs: list[float] = []
        self.values: list[float] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.episodes_since_update = 0

        self._state_history: deque[np.ndarray] = deque(maxlen=state_history_len)
        self.reset_history()

    def reset_history(self) -> None:
        """Clear the state history buffer (call at the start of each episode)."""
        self._state_history.clear()
        # Pre-fill with zeros so the first observations have a fixed shape.
        for _ in range(self.state_history_len):
            self._state_history.append(np.zeros(self.state_dim // self.state_history_len, dtype=np.float32))

    def _build_history_state(self, state: np.ndarray) -> np.ndarray:
        """Append ``state`` and return the flattened history vector."""
        self._state_history.append(state.astype(np.float32, copy=False))
        return np.concatenate(list(self._state_history), axis=0).astype(np.float32)

    def act(self, state: np.ndarray, critic_state: np.ndarray | None = None, epsilon: float | None = None) -> int:
        """Select an action for the current state (rollout).

        If ``centralized_critic`` is enabled, pass the full state vector as
        ``critic_state``; the critic will use it for value estimation while the
        actor still uses the local ``state``.
        """
        if self.state_history_len > 1:
            hist_state = self._build_history_state(state)
        else:
            hist_state = state.astype(np.float32, copy=False)

        if self.use_state_norm:
            self.state_rms.update(hist_state.reshape(1, -1))
            norm_state = self.state_rms.normalize(hist_state)
        else:
            norm_state = hist_state

        if self.centralized_critic:
            cs = critic_state.astype(np.float32, copy=False) if critic_state is not None else hist_state
            if self.use_state_norm:
                self.critic_state_rms.update(cs.reshape(1, -1))
                norm_critic_state = self.critic_state_rms.normalize(cs)
            else:
                norm_critic_state = cs
        else:
            norm_critic_state = None

        with torch.no_grad():
            state_t = torch.FloatTensor(norm_state).unsqueeze(0).to(self.device)
            critic_state_t = (
                torch.FloatTensor(norm_critic_state).unsqueeze(0).to(self.device)
                if norm_critic_state is not None else None
            )
            action, log_prob, value = self.net.act(state_t, critic_state_t)
        action = int(action.item())
        log_prob = float(log_prob.item())
        value = float(value.item())

        self.states.append(norm_state.copy())
        if self.centralized_critic:
            self.critic_states.append(norm_critic_state.copy())
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        return action

    def eval_act(self, state: np.ndarray, use_ema: bool = False) -> int:
        """Deterministic action selection for evaluation (does not store rollout data)."""
        if self.state_history_len > 1:
            hist_state = self._build_history_state(state)
        else:
            hist_state = state.astype(np.float32, copy=False)

        if self.use_state_norm:
            self.state_rms.update(hist_state.reshape(1, -1))
            norm_state = self.state_rms.normalize(hist_state)
        else:
            norm_state = hist_state

        net = self.ema_net if (use_ema and self.ema_net is not None) else self.net
        with torch.no_grad():
            state_t = torch.FloatTensor(norm_state).unsqueeze(0).to(self.device)
            logits = net.get_action_logits(state_t)
            action = logits.argmax(dim=-1)
        return int(action.item())

    def store_transition(self, reward: float, done: bool) -> None:
        """Store reward and done signal after the environment step."""
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    def shape_last_episode(self, bonus_per_step: float, length: int) -> None:
        """Add a per-step shaped bonus to the most recent ``length`` rewards.

        This implements the SRDQN-style feedback term: after an episode the
        agent receives an additional signal proportional to the average
        performance of the other supply-chain stages, encouraging policies
        that improve total system profit rather than only local profit.
        """
        if length <= 0 or bonus_per_step == 0.0:
            return
        start = max(0, len(self.rewards) - length)
        for i in range(start, len(self.rewards)):
            self.rewards[i] += bonus_per_step

    def _normalize_rewards(self) -> np.ndarray:
        rewards = np.asarray(self.rewards, dtype=np.float32)
        if self.use_reward_norm:
            self.reward_rms.update(rewards.reshape(-1, 1))
            rewards = self.reward_rms.normalize(rewards.reshape(-1, 1)).flatten()
        return rewards

    def _compute_gae(
        self,
        next_value: float,
        rewards: np.ndarray,
        next_values: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute returns and advantages using GAE.

        ``next_values`` is an optional per-transition bootstrap value array used
        by vectorized rollouts. When provided it overrides the default
        ``values[t+1]`` / ``next_value`` fallback.
        """
        values = np.asarray(self.values, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        advantages = np.zeros_like(rewards)
        last_gae = 0.0

        for t in reversed(range(len(rewards))):
            if next_values is not None:
                next_v = next_values[t]
            elif t == len(rewards) - 1:
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

    def update(
        self,
        next_state: np.ndarray | None = None,
        next_critic_state: np.ndarray | None = None,
        next_values: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Perform a PPO update using the collected rollout buffer."""
        if len(self.states) == 0:
            return {}

        self._update_lr()
        self._update_entropy()

        if next_values is not None:
            next_value = 0.0  # not used when per-transition next_values are given
        else:
            with torch.no_grad():
                if next_state is not None:
                    if self.state_history_len > 1:
                        next_state = self._build_history_state(next_state)
                    if self.use_state_norm:
                        self.state_rms.update(next_state.reshape(1, -1))
                        norm_next = self.state_rms.normalize(next_state)
                    else:
                        norm_next = next_state

                    if self.centralized_critic and next_critic_state is not None:
                        cs = next_critic_state.astype(np.float32, copy=False)
                        if self.use_state_norm:
                            self.critic_state_rms.update(cs.reshape(1, -1))
                            norm_critic_next = self.critic_state_rms.normalize(cs)
                        else:
                            norm_critic_next = cs
                        critic_next_t = torch.FloatTensor(norm_critic_next).unsqueeze(0).to(self.device)
                    else:
                        critic_next_t = None

                    next_state_t = torch.FloatTensor(norm_next).unsqueeze(0).to(self.device)
                    _, next_value = self.net.forward(next_state_t, critic_next_t)
                    next_value = float(next_value.item())
                else:
                    next_value = 0.0

        rewards = self._normalize_rewards()
        advantages, returns = self._compute_gae(next_value, rewards, next_values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = torch.FloatTensor(np.asarray(self.states)).to(self.device)
        actions = torch.LongTensor(np.asarray(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.asarray(self.log_probs)).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        old_values = torch.FloatTensor(np.asarray(self.values)).to(self.device)
        if self.centralized_critic:
            critic_states = torch.FloatTensor(np.asarray(self.critic_states)).to(self.device)
        else:
            critic_states = None

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

                log_probs, values, entropy = self.net.evaluate(batch_states, batch_actions, critic_states[batch_idx] if critic_states is not None else None)
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

                if self.use_ema:
                    with torch.no_grad():
                        for ema_p, p in zip(self.ema_net.parameters(), self.net.parameters()):
                            ema_p.data.mul_(1.0 - self.ema_tau).add_(p.data, alpha=self.ema_tau)

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_batches += 1

            if self.target_kl is not None:
                with torch.no_grad():
                    # Approximate KL(old || new) over the whole rollout.
                    _, new_log_probs, _ = self.net.evaluate(
                        states, actions, critic_states if critic_states is not None else None
                    )
                    kl = (old_log_probs - new_log_probs).mean().item()
                if kl > self.target_kl:
                    break

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
        if self.centralized_critic:
            self.critic_states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
        self.episodes_since_update = 0

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "net_state_dict": self.net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "firm_id": self.firm_id,
            "reward_rms_mean": float(np.asarray(self.reward_rms.mean).reshape(-1)[0]),
            "reward_rms_var": float(np.asarray(self.reward_rms.var).reshape(-1)[0]),
            "reward_rms_count": float(np.asarray(self.reward_rms.count).reshape(-1)[0]),
        }
        if hasattr(self, "state_rms"):
            checkpoint["state_rms_mean"] = self.state_rms.mean.copy()
            checkpoint["state_rms_var"] = self.state_rms.var.copy()
            checkpoint["state_rms_count"] = float(self.state_rms.count)
        if self.ema_net is not None:
            checkpoint["ema_net_state_dict"] = self.ema_net.state_dict()
        torch.save(checkpoint, path)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(checkpoint["net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.ema_net is not None and "ema_net_state_dict" in checkpoint:
            self.ema_net.load_state_dict(checkpoint["ema_net_state_dict"])
        self.reward_rms.mean = np.asarray(checkpoint.get("reward_rms_mean", 0.0))
        self.reward_rms.var = np.asarray(checkpoint.get("reward_rms_var", 1.0))
        self.reward_rms.count = float(checkpoint.get("reward_rms_count", 1e-4))
        if "state_rms_mean" in checkpoint and hasattr(self, "state_rms"):
            self.state_rms.mean = np.asarray(checkpoint["state_rms_mean"])
            self.state_rms.var = np.asarray(checkpoint["state_rms_var"])
            self.state_rms.count = float(checkpoint["state_rms_count"])
