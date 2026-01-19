"""Actor-Critic neural networks for PPO using Flax NNX."""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.nn.initializers import constant, orthogonal


class MLP(nnx.Module):
    """Simple MLP with tanh activations and orthogonal initialization.
    
    Attributes:
        layers: List of Dense layers.
    """

    def __init__(
        self,
        in_features: int,
        features: Sequence[int],
        final_std: float = float(jnp.sqrt(2)),
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize MLP.
        
        Args:
            in_features: Input dimension.
            features: Sequence of layer output dimensions.
            final_std: Standard deviation for orthogonal init of the final layer.
            rngs: NNX random number generators.
        """
        self.layers: list[nnx.Linear] = []
        
        layer_in_dims = [in_features] + list(features[:-1])
        for layer_in_dim, layer_out_dim in zip(layer_in_dims[:-1], features[:-1]):
            layer = nnx.Linear(
                layer_in_dim,
                layer_out_dim,
                kernel_init=orthogonal(float(jnp.sqrt(2))),
                bias_init=constant(0.0),
                rngs=rngs,
            )
            self.layers.append(layer)

        # Final layer with different initialization
        self.layers.append(
            nnx.Linear(
                layer_in_dims[-1],
                features[-1],
                kernel_init=orthogonal(final_std),
                bias_init=constant(0.0),
                rngs=rngs,
            )
        )


    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass through the MLP.

        Args:
            x: Input tensor of shape (batch, input_dim).

        Returns:
            Output tensor of shape (batch, features[-1]).
        """
        for layer in self.layers[:-1]:
            x = layer(x)
            x = jnp.tanh(x)
        x = self.layers[-1](x)
        return x


class ActorCritic(nnx.Module):
    """Combined Actor-Critic with separate networks for discrete action spaces."""

    actor: MLP
    critic: MLP
    rngs: nnx.Rngs

    def __init__(self, actor: MLP, critic: MLP, rngs: nnx.Rngs):
        """Initialize ActorCritic.
        
        Args:
            actor: MLP that takes observations (batch, obs_dim) and outputs
                   logits of shape (batch, action_dim).
            critic: MLP that takes observations (batch, obs_dim) and outputs
                    values of shape (batch, 1).
            rngs: NNX random number generators for action sampling.
        """
        self.actor = actor
        self.critic = critic
        self.rngs = rngs

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
        action: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Get action, log probability, entropy, and value.

        If action is provided, compute log_prob and entropy for that action.
        Otherwise, sample a new action using stateful rngs.

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

        if action is None:
            action = jax.random.categorical(self.rngs.action(), logits)

        log_probs = jax.nn.log_softmax(logits, axis=-1)
        log_prob = jnp.take_along_axis(log_probs, action[..., None], axis=-1).squeeze(-1)

        probs = jnp.exp(log_probs)
        entropy = -jnp.sum(probs * log_probs, axis=-1)

        value = self.critic(obs).squeeze(-1)

        return action, log_prob, entropy, value


def DefaultActorCritic(
    obs_dim: int,
    action_dim: int,
    hidden_sizes: Sequence[int] = (64, 64),
    *,
    rngs: nnx.Rngs,
) -> ActorCritic:
    """Factory function for ActorCritic with default MLP networks.

    Args:
        obs_dim: Observation dimension.
        action_dim: Number of discrete actions.
        hidden_sizes: List of hidden layer dimensions (default: [64, 64]).
        rngs: NNX random number generators.

    Returns:
        ActorCritic instance with default MLP actor and critic networks.
    """
    actor = MLP(
        obs_dim,
        (*hidden_sizes, action_dim),
        final_std=0.01,
        rngs=rngs,
    )
    critic = MLP(
        obs_dim,
        (*hidden_sizes, 1),
        final_std=1.0,
        rngs=rngs,
    )
    return ActorCritic(actor, critic, rngs)
