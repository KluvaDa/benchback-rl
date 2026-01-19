"""Actor-Critic neural networks for PPO using Flax Linen."""

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal

# Type alias for Flax parameters (nested dict structure)
ModelParams = dict[str, Any]

# Factory type: callable that returns an nn.Module
ModuleFactory = Callable[[], nn.Module]


class MLP(nn.Module):
    """Simple MLP with tanh activations and orthogonal initialization.

    Attributes:
        features: Sequence of layer output dimensions.
        final_std: Standard deviation for orthogonal init of the final layer.
    """

    features: Sequence[int]
    final_std: float = float(jnp.sqrt(2))

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass through the MLP.

        Args:
            x: Input tensor of shape (batch, input_dim).

        Returns:
            Output tensor of shape (batch, features[-1]).
        """
        for feat in self.features[:-1]:
            x = nn.Dense(feat, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.tanh(x)
        x = nn.Dense(self.features[-1], kernel_init=orthogonal(self.final_std), bias_init=constant(0.0))(x)
        return x


class ActorCritic(nn.Module):
    """Combined Actor-Critic with separate networks for discrete action spaces.

    Attributes:
        actor_factory: Factory that returns an nn.Module for the actor (obs -> logits).
        critic_factory: Factory that returns an nn.Module for the critic (obs -> value).
    """

    actor_factory: ModuleFactory
    critic_factory: ModuleFactory

    def setup(self) -> None:
        """Initialize actor and critic networks from factories."""
        self.actor = self.actor_factory()
        self.critic = self.critic_factory()

    def __call__(self, obs: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Forward pass returning both actor logits and critic value.

        Args:
            obs: Observations of shape (batch, obs_dim).

        Returns:
            Tuple of:
                - logits: Action logits of shape (batch, action_dim).
                - value: Value estimates of shape (batch,).
        """
        logits = self.actor(obs)
        value = self.critic(obs).squeeze(-1)
        return logits, value

    def get_value(self, obs: jax.Array) -> jax.Array:
        """Get value estimate for observations.

        Args:
            obs: Observations of shape (batch, obs_dim).

        Returns:
            Value estimates of shape (batch,).
        """
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(
        self,
        obs: jax.Array,
        rng_key: jax.Array,
        action: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Get action, log probability, entropy, and value.

        If action is provided, compute log_prob and entropy for that action.
        Otherwise, sample a new action.

        Args:
            obs: Observations of shape (batch, obs_dim).
            rng_key: PRNG key for sampling (ignored if action is provided).
            action: Optional actions of shape (batch,). If None, actions are sampled.

        Returns:
            Tuple of:
                - action: Actions of shape (batch,).
                - log_prob: Log probabilities of shape (batch,).
                - entropy: Entropy of the action distribution of shape (batch,).
                - value: Value estimates of shape (batch,).
        """
        logits = self.actor(obs)

        if action is None:
            # Sample action
            action = jax.random.categorical(rng_key, logits)

        log_probs = jax.nn.log_softmax(logits, axis=-1)
        log_prob = jnp.take_along_axis(log_probs, action[..., None], axis=-1).squeeze(-1)
        
        probs = jnp.exp(log_probs)
        entropy = -jnp.sum(probs * log_probs, axis=-1)

        value = self.critic(obs).squeeze(-1)

        return action, log_prob, entropy, value


def DefaultActorCritic(action_dim: int, hidden_sizes: tuple[int, ...] = (64, 64)) -> ActorCritic:
    """Factory function for ActorCritic with default MLP networks.

    Args:
        action_dim: Number of discrete actions.
        hidden_sizes: List of hidden layer dimensions (default: [64, 64]).

    Returns:
        ActorCritic instance with default MLP actor and critic networks.
    """
    return ActorCritic(
        actor_factory=partial(MLP, features=(*hidden_sizes, action_dim), final_std=0.01),
        critic_factory=partial(MLP, features=(*hidden_sizes, 1), final_std=1.0),
    )
