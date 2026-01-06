from typing import Any, cast

import jax
import jax.numpy as jnp
from flax import struct

from benchback_rl.rl_common import PPOConfig
from benchback_rl.rl_linen.models import ActorCritic, ModelParams

# Types for vmapped (batched) environments
EnvStateVmapped = Any
EnvParamsVmapped = Any
EnvStepFn = Any  # Vmapped step function: (rng, state, action, params) -> (obs, state, reward, done, info)
#                             (action,    log_prob,  entropy,   value    )
ModelActionValueResult = tuple[jax.Array, jax.Array, jax.Array, jax.Array]


@struct.dataclass
class EpisodeMetricsAccum:
    """Accumulator for episode completion metrics within a rollout.
    
    Tracks completed episodes during rollout collection. Reset at the start
    of each rollout, accumulated across steps, then converted to dict for logging.
    """
    n_completed: jax.Array = struct.field(default_factory=lambda: jnp.array(0, dtype=jnp.int32))
    sum_rewards: jax.Array = struct.field(default_factory=lambda: jnp.array(0.0))
    sum_lengths: jax.Array = struct.field(default_factory=lambda: jnp.array(0.0))
    sum_reward_per_step: jax.Array = struct.field(default_factory=lambda: jnp.array(0.0))
    min_rewards: jax.Array = struct.field(default_factory=lambda: jnp.array(jnp.inf))
    max_rewards: jax.Array = struct.field(default_factory=lambda: jnp.array(-jnp.inf))
    min_length: jax.Array = struct.field(default_factory=lambda: jnp.array(jnp.inf))
    max_length: jax.Array = struct.field(default_factory=lambda: jnp.array(-jnp.inf))

    def to_metrics_dict(self) -> dict[str, jax.Array]:
        """Compute final metrics from accumulators and return as dict for logging.
        
        Returns NaN for averages when no episodes have completed.
        """
        n = jnp.maximum(self.n_completed, 1)  # avoid division by zero
        any_completed = self.n_completed > 0
        
        # Return actual values if episodes completed, NaN otherwise
        return {
            "episodes_completed": self.n_completed,
            "avg_episode_reward": jnp.where(any_completed, self.sum_rewards / n, jnp.nan),
            "avg_episode_length": jnp.where(any_completed, self.sum_lengths / n, jnp.nan),
            "avg_reward_per_step": jnp.where(any_completed, self.sum_reward_per_step / n, jnp.nan),
            "min_episode_reward": jnp.where(any_completed, self.min_rewards, jnp.nan),
            "max_episode_reward": jnp.where(any_completed, self.max_rewards, jnp.nan),
            "min_episode_length": jnp.where(any_completed, self.min_length, jnp.nan),
            "max_episode_length": jnp.where(any_completed, self.max_length, jnp.nan),
        }


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
class RolloutWithGAE:
    """Rollout data with computed GAE advantages and returns.
    
    All arrays have shape (num_steps, num_envs, ...) except final_obs and final_value
    which have shape (num_envs, ...).
    """
    obs: jax.Array          # (num_steps, num_envs, obs_dim)
    actions: jax.Array      # (num_steps, num_envs)
    log_probs: jax.Array    # (num_steps, num_envs)
    values: jax.Array       # (num_steps, num_envs)
    rewards: jax.Array      # (num_steps, num_envs)
    dones: jax.Array        # (num_steps, num_envs)
    final_obs: jax.Array    # (num_envs, obs_dim) - obs after final step
    final_value: jax.Array  # (num_envs,) - V(final_obs) for GAE computation
    advantages: jax.Array   # (num_steps, num_envs)
    returns: jax.Array      # (num_steps, num_envs)


def rollout_with_gae(
    config: PPOConfig,
    env_step_fn: EnvStepFn,
    env_params: EnvParamsVmapped,
    env_carry: EnvCarry,
    model: ActorCritic,
    model_params: ModelParams,
    rng_key: jax.Array,
) -> tuple[RolloutWithGAE, EnvCarry, dict[str, jax.Array]]:
    """Collect a rollout and compute GAE advantages.
    
    Main entry point for the rollout phase. Combines environment interaction
    with advantage estimation.
    
    Args:
        config: PPO configuration with num_steps, num_envs, gamma, gae_lambda.
        env_step_fn: Vmapped environment step function.
        env_params: Environment parameters.
        env_carry: Persistent state (obs, env_state, episode trackers).
        model: ActorCritic Linen module.
        model_params: Model parameters for model.apply.
        rng_key: PRNG key for action sampling and env steps.
    
    Returns:
        Tuple of (RolloutWithGAE, updated EnvCarry, episode_metrics dict for logging).
        Episode metrics contain NaN values if no episodes completed this rollout.
    """
    # Collect rollout data
    (obs, actions, log_probs, values, rewards, dones, final_obs, final_value,
     new_env_carry, episode_metrics) = collect_rollout(
        config, env_step_fn, env_params, env_carry, model, model_params, rng_key
    )
    
    # Compute GAE
    advantages, returns = compute_gae(
        values, rewards, dones, final_value, config.gamma, config.gae_lambda
    )
    
    rollout = RolloutWithGAE(
        obs=obs,
        actions=actions,
        log_probs=log_probs,
        values=values,
        rewards=rewards,
        dones=dones,
        final_obs=final_obs,
        final_value=final_value,
        advantages=advantages,
        returns=returns,
    )
    
    return rollout, new_env_carry, episode_metrics


def collect_rollout(
    config: PPOConfig,
    env_step_fn: EnvStepFn,
    env_params: EnvParamsVmapped,
    env_carry: EnvCarry,
    model: ActorCritic,
    model_params: ModelParams,
    rng_key: jax.Array,
) -> tuple[
    jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array,  # obs, actions, log_probs, values, rewards, dones
    jax.Array, jax.Array,  # final_obs, final_value
    EnvCarry, dict[str, jax.Array]  # env_carry, episode_metrics
]:
    """Collect a rollout of experience from vectorized environments.
    
    Uses jax.lax.scan for efficient, jittable rollout collection.
    
    Args:
        config: PPO configuration with num_steps and num_envs.
        env_step_fn: Vmapped environment step function.
        env_params: Environment parameters.
        env_carry: Persistent state (obs, env_state, episode trackers).
        model: ActorCritic Linen module.
        model_params: Model parameters for model.apply.
        rng_key: PRNG key for action sampling and env steps.
    
    Returns:
        Tuple of (obs, actions, log_probs, values, rewards, dones,
                  final_obs, final_value, updated EnvCarry, episode_metrics).
    """
    
    def step(carry, _):
        """Single step of rollout collection."""
        env_carry_, metrics_accum_, rng_key_ = carry
        
        rng_key_, rng_key_action, rng_key_step = jax.random.split(rng_key_, 3)
        
        # Apply model to get action, log_prob, and value
        model_action_value_result = model.apply(model_params,
                                                env_carry_.obs,
                                                rng_key_action,
                                                method=model.get_action_and_value)
        action, log_prob, _, value = cast(ModelActionValueResult, model_action_value_result)
        
        # Step environment (vmapped) - split key for each parallel environment
        rng_keys_step = jax.random.split(rng_key_step, config.num_envs)
        next_obs, next_env_state, reward, done, _ = env_step_fn(rng_keys_step, env_carry_.env_state, action, env_params)
        
        # Update episode trackers
        episode_rewards = env_carry_.episode_rewards + reward
        episode_lengths = env_carry_.episode_lengths + 1
        
        # Update metrics accumulators for completed episodes
        next_metrics_accum = EpisodeMetricsAccum(
            n_completed=metrics_accum_.n_completed + done.sum().astype(jnp.int32),
            sum_rewards=metrics_accum_.sum_rewards + (episode_rewards * done).sum(),
            sum_lengths=metrics_accum_.sum_lengths + (episode_lengths * done).sum(),
            sum_reward_per_step=metrics_accum_.sum_reward_per_step + (episode_rewards / episode_lengths * done).sum(),
            min_rewards=jnp.minimum(metrics_accum_.min_rewards,
                                    jnp.min(jnp.asarray(jnp.where(done, episode_rewards, jnp.inf)))),
            max_rewards=jnp.maximum(metrics_accum_.max_rewards,
                                    jnp.max(jnp.asarray(jnp.where(done, episode_rewards, -jnp.inf)))),
            min_length=jnp.minimum(metrics_accum_.min_length,
                                   jnp.min(jnp.asarray(jnp.where(done, episode_lengths, jnp.inf)))),
            max_length=jnp.maximum(metrics_accum_.max_length,
                                   jnp.max(jnp.asarray(jnp.where(done, episode_lengths, -jnp.inf)))),
        )
        
        # Update EnvCarry (reset episode trackers on done)
        next_env_carry = EnvCarry(
            obs=next_obs,
            env_state=next_env_state,
            episode_rewards=jnp.asarray(jnp.where(done, 0.0, episode_rewards)),
            episode_lengths=jnp.asarray(jnp.where(done, 0, episode_lengths)),
        )

        # Store transition data
        transition = (env_carry_.obs, action, log_prob, value, reward, done)
        
        return (next_env_carry, next_metrics_accum, rng_key_), transition
    
    # Initialize fresh metrics accumulators for this rollout
    metrics_accum = EpisodeMetricsAccum()
    
    # Run scan over num_steps
    (final_env_carry, final_metrics_accum, _), transitions = jax.lax.scan(
        step,
        init=(env_carry, metrics_accum, rng_key),
        xs=None,
        length=config.num_steps,
    )
    
    # Unpack transitions: each has shape (num_steps, num_envs, ...)
    obs, actions, log_probs, values, rewards, dones = transitions
    
    # Compute value for the observation after the final step (for GAE)
    final_obs = final_env_carry.obs
    final_value = jnp.asarray(model.apply(model_params, final_obs, method=model.get_value))
    
    # Convert accumulators to metrics dict
    episode_metrics = final_metrics_accum.to_metrics_dict()
    
    return (obs, actions, log_probs, values, rewards, dones,
            final_obs, final_value, final_env_carry, episode_metrics)


def compute_gae(
    values: jax.Array,
    rewards: jax.Array,
    dones: jax.Array,
    final_value: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """Compute Generalized Advantage Estimation (GAE).
    
    Args:
        values: Value estimates, shape (num_steps, num_envs).
        rewards: Rewards, shape (num_steps, num_envs).
        dones: Done flags, shape (num_steps, num_envs).
        final_value: Value of final observation, shape (num_envs,).
        gamma: Discount factor.
        gae_lambda: GAE lambda parameter.
    
    Returns:
        Tuple of (advantages, returns), each shape (num_steps, num_envs).
    """
    def gae_step(carry, xs):
        # "next" refers to timestep t+1, not the next iteration (in reverse order)
        next_gae, next_value = carry
        value, reward, done = xs
        
        not_done = 1.0 - done
        delta = reward + gamma * next_value * not_done - value
        gae = delta + gamma * gae_lambda * not_done * next_gae
        
        return (gae, value), gae
    
    # Initialize: gae beyond horizon is 0, next_value is V(final_obs)
    num_envs = final_value.shape[0]
    init_carry = (jnp.zeros(num_envs), final_value)
    
    _, advantages = jax.lax.scan(
        gae_step,
        init_carry,
        (values, rewards, dones),
        reverse=True,
    )
    
    # TD(lambda) returns: advantages + values
    returns = advantages + values
    
    return advantages, returns
