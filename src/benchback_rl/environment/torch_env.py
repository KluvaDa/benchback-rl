"""PyTorch environment wrapper using JAX gymnax under the hood."""

from typing import Any

import gymnax
import jax
import jax.numpy as jnp
import torch
import torch.utils.dlpack


class TorchEnv:
    """Vectorized PyTorch environment wrapper using gymnax internally."""

    def __init__(self, env_name: str, num_envs: int, jit: bool = True) -> None:
        env, self.params = gymnax.make(env_name)

        self.num_envs = num_envs
        self.num_actions = env.num_actions

        # vmap over: keys (num_envs,), state (num_envs, ...), action (num_envs, ...)
        # params are shared (not vmapped)
        self._reset_fn = jax.vmap(env.reset, in_axes=(0, None))
        self._step_fn = jax.vmap(env.step, in_axes=(0, 0, 0, None))

        if jit:
            self._reset_fn = jax.jit(self._reset_fn)
            self._step_fn = jax.jit(self._step_fn)

    def reset(self, key: jax.Array) -> tuple[torch.Tensor, Any]:
        """Reset all environments."""
        # keys: (num_envs, 2) - one key per environment
        keys = jax.random.split(key, self.num_envs)
        obs, state = self._reset_fn(keys, self.params)
        return jax_to_torch(obs), state

    def step(
        self, key: jax.Array, state: Any, action: torch.Tensor
    ) -> tuple[torch.Tensor, Any, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Step all environments."""
        # keys: (num_envs, 2) - one key per environment
        # state: (num_envs, ...) - batched env state
        # action: (num_envs,) discrete or (num_envs, action_dim) continuous
        keys = jax.random.split(key, self.num_envs)
        obs, state, reward, done, info = self._step_fn(
            keys, state, torch_to_jax(action), self.params
        )
        return (
            jax_to_torch(obs),
            state,
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
