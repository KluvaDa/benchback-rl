"""PyTorch environment wrapper using JAX gymnax under the hood."""

from typing import Any

import gymnax
from gymnax.environments import spaces
import jax
import jax.numpy as jnp
import torch
import torch.utils.dlpack


class TorchEnv:
    """Vectorized PyTorch environment wrapper using gymnax internally.

    Only supports environments with:
    - Discrete action space
    - 1D Box observation space (no image observations)

    The internal JAX env state is managed by this class and not exposed.
    """

    def __init__(self, env_name: str, num_envs: int, jit: bool = True) -> None:
        self.params: Any
        self.num_actions: int
        self.obs_dim: int
        self.num_envs: int
        self._state: Any = None  # Internal JAX env state

        env, self.params = gymnax.make(env_name)

        # Validate and extract action space info
        action_space = env.action_space(self.params)
        if not isinstance(action_space, spaces.Discrete):
            raise ValueError(f"Only Discrete action spaces supported, got {type(action_space).__name__}")
        self.num_actions = action_space.n

        # Validate and extract observation space info
        obs_space = env.observation_space(self.params)
        if not isinstance(obs_space, spaces.Box):
            raise ValueError(f"Only Box observation spaces supported, got {type(obs_space).__name__}")
        if len(obs_space.shape) != 1:
            raise ValueError(f"Only 1D observations supported (no images), got shape {obs_space.shape}")
        self.obs_dim = obs_space.shape[0]

        self.num_envs = num_envs

        # vmap over: keys (num_envs,), state (num_envs, ...), action (num_envs, ...)
        # params are shared (not vmapped)
        self._reset_fn = jax.vmap(env.reset, in_axes=(0, None))
        self._step_fn = jax.vmap(env.step, in_axes=(0, 0, 0, None))

        if jit:
            self._reset_fn = jax.jit(self._reset_fn)
            self._step_fn = jax.jit(self._step_fn)

    def reset(self, key: jax.Array) -> torch.Tensor:
        """Reset all environments.

        Returns:
            Initial observations (num_envs, obs_dim)
        """
        # keys: (num_envs, 2) - one key per environment
        keys = jax.random.split(key, self.num_envs)
        obs, self._state = self._reset_fn(keys, self.params)
        return jax_to_torch(obs)

    def step(
        self, key: jax.Array, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Step all environments.

        Returns:
            Tuple of (obs, reward, done, info)
        """
        assert self._state is not None, "Must call reset() before step()"
        # keys: (num_envs, 2) - one key per environment
        # action: (num_envs,) discrete or (num_envs, action_dim) continuous
        keys = jax.random.split(key, self.num_envs)
        obs, self._state, reward, done, info = self._step_fn(
            keys, self._state, torch_to_jax(action), self.params
        )
        return (
            jax_to_torch(obs),
            jax_to_torch(reward),
            jax_to_torch(done),
            jax_pytree_to_torch(info),
        )


def jax_to_torch(x: jax.Array) -> torch.Tensor:
    """Convert JAX array to PyTorch CUDA tensor via DLPack."""
    return torch.utils.dlpack.from_dlpack(x)


def torch_to_jax(x: torch.Tensor) -> jax.Array:
    """Convert PyTorch tensor to JAX array via DLPack."""
    return jnp.from_dlpack(x)


def jax_pytree_to_torch(tree: Any) -> Any:
    """Convert a pytree of JAX arrays to PyTorch tensors. Non-arrays pass through."""
    return jax.tree.map(
        lambda x: jax_to_torch(x) if isinstance(x, jax.Array) else x, tree
    )
