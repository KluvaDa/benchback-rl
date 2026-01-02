
from typing import Any, cast

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
import optax
import time

from benchback_rl.rl_common import PPOConfig
from benchback_rl.rl_linen.models import ActorCritic, ModelParams

# Types for vmapped (batched) environments
EnvStateVmapped = Any
EnvParamsVmapped = Any
EnvVmapped = Any
#                             (action,    log_prob,  entropy,   value    )
ModelActionValueResult = tuple[jax.Array, jax.Array, jax.Array, jax.Array]


@struct.dataclass
class RolloutMetricsCarry:
    """Accumulators for episode metrics within a rollout.
    
    Reset at the start of each rollout, accumulated across steps.
    Used internally; converted to RolloutMetrics for output.
    """
    n_completed: jax.Array = struct.field(default_factory=lambda: jnp.array(0, dtype=jnp.int32))
    sum_rewards: jax.Array = struct.field(default_factory=lambda: jnp.array(0.0))
    sum_lengths: jax.Array = struct.field(default_factory=lambda: jnp.array(0.0))
    sum_reward_per_step: jax.Array = struct.field(default_factory=lambda: jnp.array(0.0))
    min_rewards: jax.Array = struct.field(default_factory=lambda: jnp.array(jnp.inf))
    max_rewards: jax.Array = struct.field(default_factory=lambda: jnp.array(-jnp.inf))
    min_length: jax.Array = struct.field(default_factory=lambda: jnp.array(jnp.inf))
    max_length: jax.Array = struct.field(default_factory=lambda: jnp.array(-jnp.inf))

    def to_metrics(self) -> "RolloutMetrics":
        """Convert accumulators to final metrics (averages, etc.)."""
        n = jnp.maximum(self.n_completed, 1)  # avoid division by zero
        return RolloutMetrics(
            episodes_completed=self.n_completed,
            avg_episode_reward=self.sum_rewards / n,
            avg_episode_length=self.sum_lengths / n,
            avg_reward_per_step=self.sum_reward_per_step / n,
            min_episode_reward=self.min_rewards,
            max_episode_reward=self.max_rewards,
            min_episode_length=self.min_length,
            max_episode_length=self.max_length,
        )


@struct.dataclass
class RolloutMetrics:
    """Episode metrics for logging.
    
    Contains computed averages and statistics from completed episodes.
    """
    episodes_completed: jax.Array = struct.field(default_factory=lambda: jnp.array(0, dtype=jnp.int32))
    avg_episode_reward: jax.Array = struct.field(default_factory=lambda: jnp.array(jnp.nan))
    avg_episode_length: jax.Array = struct.field(default_factory=lambda: jnp.array(jnp.nan))
    avg_reward_per_step: jax.Array = struct.field(default_factory=lambda: jnp.array(jnp.nan))
    min_episode_reward: jax.Array = struct.field(default_factory=lambda: jnp.array(jnp.nan))
    max_episode_reward: jax.Array = struct.field(default_factory=lambda: jnp.array(jnp.nan))
    min_episode_length: jax.Array = struct.field(default_factory=lambda: jnp.array(jnp.nan))
    max_episode_length: jax.Array = struct.field(default_factory=lambda: jnp.array(jnp.nan))


@struct.dataclass
class EnvCarry:
    """Persistent state carried across rollouts.
    
    Bundles environment state and per-episode trackers.
    """
    # Environment state
    obs: jax.Array              # (num_envs, obs_dim) - current observation
    env_state: EnvStateVmapped  # Vmapped environment state
    
    # Per-episode accumulators (persist across rollouts, reset when episode completes)
    episode_rewards: jax.Array  # (num_envs,) - accumulated rewards for current episode
    episode_lengths: jax.Array  # (num_envs,) - accumulated lengths for current episode

    @classmethod
    def from_reset(cls, obs: jax.Array, env_state: EnvStateVmapped) -> "EnvCarry":
        """Create EnvCarry from environment reset, initializing all trackers."""
        num_envs = obs.shape[0]
        return cls(
            obs=obs,
            env_state=env_state,
            episode_rewards=jnp.zeros(num_envs),
            episode_lengths=jnp.zeros(num_envs, dtype=jnp.int32),
        )


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
    final_env_state: EnvStateVmapped  # Final env state after rollout - for bootstrap value


def _collect_rollout(
    config: PPOConfig,
    env: EnvVmapped,
    env_params: EnvParamsVmapped,
    env_carry: EnvCarry,
    last_metrics: RolloutMetrics,
    model: ActorCritic,
    model_params: ModelParams,
    rng_key: jax.Array,
) -> tuple[Rollout, EnvCarry, RolloutMetrics]:
    """Collect a rollout of experience from vectorized environments.
    
    Uses jax.lax.scan for efficient, jittable rollout collection.
    
    Args:
        config: PPO configuration with num_steps and num_envs.
        env: Vmapped Gymnax environment.
        env_params: Vmapped environment parameters.
        env_carry: Persistent state (obs, env_state, episode trackers).
        last_metrics: Fallback metrics from most recent rollout where episodes completed.
        model: ActorCritic Linen module.
        model_params: Model parameters for model.apply.
        rng_key: PRNG key for action sampling and env steps.
    
    Returns:
        Tuple of (Rollout, updated EnvCarry, RolloutMetrics to log/use as next fallback).
    """
    
    def _step(carry, _):  # (carry, input) - no input because no xs
        """Single step of rollout collection."""
        env_carry_, metrics_carry_, rng_key_ = carry
        
        # rng 
        rng_key_, rng_key_action, rng_key_step = jax.random.split(rng_key_, 3)
        
        # Apply model to get action, log_prob, and value
        model_action_value_result = model.apply(model_params,
                                                env_carry_.obs,
                                                rng_key_action,
                                                method=model.get_action_and_value)
        action, log_prob, _, value = cast(ModelActionValueResult, model_action_value_result)
        
        # Step environment (vmapped)
        next_obs, next_env_state, reward, done, _ = env.step(rng_key_step, env_carry_.env_state, action, env_params)
        

        ##### METRICS #####

        episode_rewards = env_carry_.episode_rewards + reward
        episode_lengths = env_carry_.episode_lengths + 1
        
        # Update metrics accumulators for completed episodes
        # jnp.asarray used to help Pylance with type inference (jnp.where returns union type)
        next_metrics_carry = RolloutMetricsCarry(
            n_completed=metrics_carry_.n_completed + done.sum().astype(jnp.int32),
            sum_rewards=metrics_carry_.sum_rewards + (episode_rewards * done).sum(),
            sum_lengths=metrics_carry_.sum_lengths + (episode_lengths * done).sum(),
            sum_reward_per_step=metrics_carry_.sum_reward_per_step + (episode_rewards / episode_lengths * done).sum(),
            min_rewards=jnp.minimum(metrics_carry_.min_rewards,
                                    jnp.min(jnp.asarray(jnp.where(done, episode_rewards, jnp.inf)))),
            max_rewards=jnp.maximum(metrics_carry_.max_rewards,
                                    jnp.max(jnp.asarray(jnp.where(done, episode_rewards, -jnp.inf)))),
            min_length=jnp.minimum(metrics_carry_.min_length,
                                   jnp.min(jnp.asarray(jnp.where(done, episode_lengths, jnp.inf)))),
            max_length=jnp.maximum(metrics_carry_.max_length,
                                   jnp.max(jnp.asarray(jnp.where(done, episode_lengths, -jnp.inf)))),
        )
        
        # Update EnvCarry
        # jnp.asarray used to help Pylance with type inference
        next_env_carry = EnvCarry(
            obs=next_obs,
            env_state=next_env_state,
            episode_rewards=jnp.asarray(jnp.where(done, 0.0, episode_rewards)),
            episode_lengths=jnp.asarray(jnp.where(done, 0, episode_lengths)),
        )

        ##### END METRICS #####

        # Store transition data
        transition = (env_carry_.obs, action, log_prob, value, reward, done)
        
        return (next_env_carry, next_metrics_carry, rng_key_), transition
    
    # Initialize fresh metrics accumulators for this rollout
    metrics_carry = RolloutMetricsCarry()
    
    # Run scan over num_steps
    (final_env_carry, final_metrics_carry, _), transitions = jax.lax.scan(
        _step,
        init=(env_carry, metrics_carry, rng_key),
        xs=None,
        length=config.num_steps,
    )
    
    # Unpack transitions: each has shape (num_steps, num_envs, ...)
    obs, actions, log_probs, values, rewards, dones = transitions
    
    rollout = Rollout(
        obs=obs,
        actions=actions,
        log_probs=log_probs,
        values=values,
        rewards=rewards,
        dones=dones,
        final_obs=final_env_carry.obs,
        final_env_state=final_env_carry.env_state
    )
    
    # Convert accumulators to metrics, use fallback if no episodes completed
    any_completed = final_metrics_carry.n_completed > 0
    output_metrics = jax.lax.cond(
        any_completed,
        lambda: final_metrics_carry.to_metrics(),
        lambda: last_metrics,
    )
    
    return rollout, final_env_carry, output_metrics

