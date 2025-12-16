"""Actor-Critic neural networks for PPO."""

import numpy as np
import torch
import torch.nn as nn


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal initialization for linear layers."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """Combined Actor-Critic with separate networks for discrete action spaces."""

    def __init__(self, actor: nn.Module, critic: nn.Module) -> None:
        """
        Args:
            actor: nn.Module that takes observations (batch, obs_dim) and outputs logits of shape (batch, action_dim).
            critic: nn.Module that takes observations (batch, obs_dim) and outputs values of shape (batch, 1).
        """
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.cuda()  # Always on CUDA

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value estimate for observations.

        Args:
            obs: Observations of shape (batch, obs_dim).

        Returns:
            Value estimates of shape (batch,).
        """
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log probability, entropy, and value.

        If action is provided, compute log_prob and entropy for that action. Otherwise, sample a new action.

        Args:
            obs: Observations of shape (batch, obs_dim).
            action: Optional actions of shape (batch,). If None, actions are sampled.

        Returns:
            Tuple of:
                - action: Actions of shape (batch,).
                - log_prob: Log probabilities of shape (batch,).
                - entropy: Entropy of the action distribution of shape (batch,).
                - value: Value estimates of shape (batch,).
        """
        logits = self.actor(obs)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(obs).squeeze(-1)
        return action, log_prob, entropy, value


class DefaultActorCritic(ActorCritic):
    """ActorCritic with default MLP networks for discrete action spaces."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
        )
        critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        super().__init__(actor, critic)
