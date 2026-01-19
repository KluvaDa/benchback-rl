"""Actor-Critic neural networks for PPO."""

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal initialization for linear layers."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class MLP(nn.Module):
    """Multi-layer perceptron with orthogonal initialization."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_sizes: Sequence[int],
        output_std: float = np.sqrt(2),
    ) -> None:
        """
        Args:
            in_dim: Input dimension.
            out_dim: Output dimension.
            hidden_sizes: Sequence[int] of hidden layer sizes.
            output_std: Standard deviation for output layer initialization.
        """
        super().__init__()
        layers = []
        current_dim = in_dim
        for hidden_dim in hidden_sizes:
            layers.append(layer_init(nn.Linear(current_dim, hidden_dim)))
            layers.append(nn.Tanh())
            current_dim = hidden_dim
        layers.append(layer_init(nn.Linear(current_dim, out_dim), std=output_std))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: Sequence[int] = (64, 64)) -> None:
        actor = MLP(obs_dim, action_dim, hidden_sizes, output_std=0.01)
        critic = MLP(obs_dim, 1, hidden_sizes, output_std=1.0)
        super().__init__(actor, critic)
