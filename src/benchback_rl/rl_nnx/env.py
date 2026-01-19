"""Stateful vectorized environment wrapper using Flax NNX.

Wraps a gymnax environment to provide a stateful interface similar to 
gymnasium's VectorEnv, while maintaining NNX graph tracking for JIT compilation.
"""

from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import nnx
import gymnax
from gymnax.environments import spaces

# Type aliases for gymnax environments
Env = Any
EnvParams = Any

class EnvState(nnx.Variable):
    pass

class EnvStaticVariable(nnx.Variable):
    """This is only a temporary workaround until flax.nnx.static is available in version 0.12 onwards."""
    pass

class NnxVecEnv(nnx.Module):
    """Stateful vectorized environment wrapper.
    
    Wraps a gymnax environment and manages environment state internally,
    providing a cleaner interface for RL algorithms.

    rngs() will be used with the default collection stream.
    
    Attributes:
        obs: Current observations, shape (num_envs, obs_dim).
        state: Current environment states (vmapped gymnax EnvState).
        episode_rewards: Cumulative rewards for current episodes.
        episode_lengths: Current episode lengths.
    """
    
    def __init__(
        self,
        env_name: str,
        num_envs: int,
        rngs: nnx.Rngs,
    ) -> None:

        """Initialize the vectorized environment.

        rngs() will be used with the default collection stream.

        Args:
            env_name: Name of the gymnax environment.
            num_envs: Number of parallel environments.
            rngs: NNX Rngs object for random number generation. using rngs.env()
        """
        # Declaring stateful variables
        self.state: EnvState
        self.rngs: nnx.Rngs = rngs
        self.env_params: EnvStaticVariable 


        # Declaring static variables
        self.env_name: str = env_name
        self.num_envs: int = num_envs
        self.env: Any
        self.num_actions: int
        self.obs_dim: int
        self.reset_fn: Callable
        self.step_fn: Callable

        # Gymnax environment
        self.env, env_params = gymnax.make(env_name)
        self.env_params = EnvStaticVariable(env_params)  # Temporary workaround for static variable

        # Validate and extract action space info
        action_space = self.env.action_space(self.env_params.value)
        if not isinstance(action_space, spaces.Discrete):
            raise ValueError(f"Only Discrete action spaces supported, got {type(action_space).__name__}")
        self.num_actions = action_space.n

        # Validate and extract observation space info
        obs_space = self.env.observation_space(self.env_params.value)
        if not isinstance(obs_space, spaces.Box):
            raise ValueError(f"Only Box observation spaces supported, got {type(obs_space).__name__}")
        if len(obs_space.shape) != 1:
            raise ValueError(f"Only 1D observations supported (no images), got shape {obs_space.shape}")
        self.obs_dim = obs_space.shape[0]

        # Initialize state variable (will be properly set during reset())
        self.state = EnvState(None)

        # vmap over: keys (num_envs,), state (num_envs, ...), action (num_envs, ...)
        # params are shared (not vmapped)
        # Note: not JIT compiled here - compiled from above in the rollout
        self.reset_fn = jax.vmap(self.env.reset, in_axes=(0, None))
        self.step_fn = jax.vmap(self.env.step, in_axes=(0, 0, 0, None))
    
    def reset(self) -> jax.Array:
        """Reset all environments.
        
        Returns:
            Initial observations, shape (num_envs, obs_dim).
        """
        reset_keys = jax.random.split(self.rngs.env(), self.num_envs)
        obs, self.state.value = self.reset_fn(reset_keys, self.env_params.value)
        
        return obs
    
    def step(self, actions: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, Any]]:
        """Step all environments.
        
        Args:
            actions: Actions for each environment, shape (num_envs,).
        
        Returns:
            Tuple of (obs, rewards, dones, info) where:
                - obs: New observations, shape (num_envs, obs_dim)
                - rewards: Rewards, shape (num_envs,)
                - dones: Done flags, shape (num_envs,)
                - info: Dict with episode info (rewards/lengths for completed episodes)
        """
        step_keys = jax.random.split(self.rngs.env(), self.num_envs)
        
        next_obs, next_state, rewards, dones, info = self.step_fn(
            step_keys, self.state.value, actions, self.env_params.value
        )
        self.state.value = next_state
        
        return next_obs, rewards, dones, info
    