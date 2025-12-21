
from typing import Any

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
import optax
import time

from benchback_rl.rl_common import PPOConfig
from benchback_rl.rl_linen.models import ActorCritic, Params

# Types for vmapped (batched) environments
EnvStateVmapped = Any
EnvParamsVmapped = Any
EnvVmapped = Any


@struct.dataclass
class Rollout:
    """Rollout data collected from environment interactions.
    
    All arrays have shape (num_steps, num_envs, ...) except final_obs
    which has shape (num_envs, obs_dim).
    """
    obs: jax.Array          # (num_steps, num_envs, obs_dim)
    actions: jax.Array      # (num_steps, num_envs)
    log_probs: jax.Array    # (num_steps, num_envs)
    values: jax.Array       # (num_steps, num_envs)
    rewards: jax.Array      # (num_steps, num_envs)
    dones: jax.Array        # (num_steps, num_envs)
    final_obs: jax.Array    # (num_envs, obs_dim) - for bootstrap value


def _collect_rollout(
    config: PPOConfig,
    env: EnvVmapped,
    env_params: EnvParamsVmapped,
    env_state: EnvStateVmapped,
    model: ActorCritic,
    params: Params,
    rng_key: jax.Array,
) -> tuple[Rollout, EnvStateVmapped]:
    """Collect a rollout of experience from vectorized environments.
    
    Uses jax.lax.scan for efficient, jittable rollout collection.
    
    Args:
        config: PPO configuration with num_steps and num_envs.
        env: Vmapped Gymnax environment.
        env_params: Vmapped environment parameters.
        env_state: Current vmapped environment state (num_envs,).
        model: ActorCritic Linen module.
        params: Model parameters for model.apply.
        rng_key: PRNG key for action sampling and env steps.
    
    Returns:
        Tuple of (Rollout dataclass, final env_state after rollout).
    """
    
    def _step(carry, _):
        """Single step of rollout collection."""
        env_state, rng_key = carry
        
        # Split keys for action sampling and env step
        rng_key, key_action, key_step = jax.random.split(rng_key, 3)
        
        # Get current observation from env state
        obs = env_state.obs  # (num_envs, obs_dim)
        
        # Get action, log_prob, and value from model
        action: jax.Array
        log_prob: jax.Array
        value: jax.Array
        action, log_prob, _, value = model.apply(  # pyright: ignore[reportAssignmentType]
            params, obs, key_action, method=model.get_action_and_value
        )
        
        # Step environment (vmapped)
        next_obs, next_env_state, reward, done, _ = env.step(key_step, env_state, action, env_params)
        
        # Store transition data
        transition = (obs, action, log_prob, value, reward, done)
        
        return (next_env_state, rng_key), transition
    
    # Run scan over num_steps
    (final_env_state, _), transitions = jax.lax.scan(
        _step,
        init=(env_state, rng_key),
        xs=None,
        length=config.num_steps,
    )
    
    # Unpack transitions: each has shape (num_steps, num_envs, ...)
    obs, actions, log_probs, values, rewards, dones = transitions
    
    # Get final observation for bootstrap value
    final_obs = final_env_state.obs
    
    rollout = Rollout(
        obs=obs,
        actions=actions,
        log_probs=log_probs,
        values=values,
        rewards=rewards,
        dones=dones,
        final_obs=final_obs,
    )
    
    return rollout, final_env_state

