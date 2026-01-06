from typing import cast

import jax
import jax.numpy as jnp
from flax import struct
import optax

from benchback_rl.rl_common import PPOConfig
from benchback_rl.rl_linen.models import ActorCritic, ModelParams
from benchback_rl.rl_linen.ppo.rollout import RolloutWithGAE

#                             (action,    log_prob,  entropy,   value    )
ModelActionValueResult = tuple[jax.Array, jax.Array, jax.Array, jax.Array]


@struct.dataclass
class TrainState:
    """Training state bundling model parameters and optimizer state.
    
    Attributes:
        model_params: Flax model parameters.
        opt_state: Optax optimizer state.
        step: Current optimizer step count (for LR scheduling).
    """
    model_params: ModelParams
    opt_state: optax.OptState
    step: jnp.ndarray = struct.field(default_factory=lambda: jnp.array(0, dtype=jnp.int32))


@struct.dataclass
class UpdateMetricsAccum:
    """Accumulator for update phase metrics (running sums).
    
    Accumulated across all minibatch updates within the update phase,
    then divided by n_updates to get means for logging.
    """
    sum_total_loss: jax.Array = struct.field(default_factory=lambda: jnp.array(0.0))
    sum_policy_loss: jax.Array = struct.field(default_factory=lambda: jnp.array(0.0))
    sum_value_loss: jax.Array = struct.field(default_factory=lambda: jnp.array(0.0))
    sum_entropy_loss: jax.Array = struct.field(default_factory=lambda: jnp.array(0.0))
    sum_approx_kl: jax.Array = struct.field(default_factory=lambda: jnp.array(0.0))
    sum_clip_frac: jax.Array = struct.field(default_factory=lambda: jnp.array(0.0))
    n_updates: jax.Array = struct.field(default_factory=lambda: jnp.array(0, dtype=jnp.int32))

    def to_metrics_dict(self) -> dict[str, jax.Array]:
        """Compute final metrics from accumulators and return as dict for logging."""
        n = self.n_updates.astype(jnp.float32)
        return {
            "total_loss": self.sum_total_loss / n,
            "policy_loss": self.sum_policy_loss / n,
            "value_loss": self.sum_value_loss / n,
            "entropy_loss": self.sum_entropy_loss / n,
            "approx_kl": self.sum_approx_kl / n,
            "clip_frac": self.sum_clip_frac / n,
        }


@struct.dataclass
class UpdateCarry:
    """Carry state for the update phase scan loops."""
    train_state: TrainState
    metrics_accum: UpdateMetricsAccum
    rng_key: jax.Array


@struct.dataclass
class Minibatch:
    """A minibatch of flattened rollout data for PPO update."""
    obs: jax.Array          # (minibatch_size, obs_dim)
    actions: jax.Array      # (minibatch_size,)
    log_probs: jax.Array    # (minibatch_size,)
    advantages: jax.Array   # (minibatch_size,)
    returns: jax.Array      # (minibatch_size,)
    values: jax.Array       # (minibatch_size,)


def flatten_and_shuffle(
    rollout_with_gae: RolloutWithGAE,
    rng_key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Flatten rollout data and return shuffled indices.
    
    Args:
        rollout_with_gae: Rollout data with computed advantages and returns.
        rng_key: PRNG key for shuffling.
    
    Returns:
        Tuple of flattened arrays: (obs, actions, log_probs, advantages, returns, values)
        all with shape (batch_size, ...) and shuffled along axis 0.
    """
    batch_size = rollout_with_gae.obs.shape[0] * rollout_with_gae.obs.shape[1]
    obs_dim = rollout_with_gae.obs.shape[2]
    
    # Flatten: (num_steps, num_envs, ...) -> (batch_size, ...)
    flat_obs = rollout_with_gae.obs.reshape(batch_size, obs_dim)
    flat_actions = rollout_with_gae.actions.reshape(batch_size)
    flat_log_probs = rollout_with_gae.log_probs.reshape(batch_size)
    flat_advantages = rollout_with_gae.advantages.reshape(batch_size)
    flat_returns = rollout_with_gae.returns.reshape(batch_size)
    flat_values = rollout_with_gae.values.reshape(batch_size)
    
    # Shuffle
    indices = jax.random.permutation(rng_key, batch_size)
    
    return (
        flat_obs[indices],
        flat_actions[indices],
        flat_log_probs[indices],
        flat_advantages[indices],
        flat_returns[indices],
        flat_values[indices],
    )


def get_minibatches(
    rollout_with_gae: RolloutWithGAE,
    num_minibatches: int,
    rng_key: jax.Array,
) -> Minibatch:
    """Prepare minibatches from rollout data.
    
    Args:
        rollout_with_gae: Rollout data with computed advantages and returns.
        num_minibatches: Number of minibatches to split the data into.
        rng_key: PRNG key for shuffling.
    
    Returns:
        Minibatch with arrays of shape (num_minibatches, minibatch_size, ...).
    """
    flat_data = flatten_and_shuffle(rollout_with_gae, rng_key)
    obs, actions, log_probs, advantages, returns, values = flat_data
    
    # Reshape into minibatches: (batch_size, ...) -> (num_minibatches, minibatch_size, ...)
    # C-order (row-major) layout ensures trailing dimensions stay contiguous:
    # elements [i, j, ...] map to [i // minibatch_size, i % minibatch_size, ...]
    batch_size = obs.shape[0]
    minibatch_size = batch_size // num_minibatches
    obs_dim = obs.shape[1]
    
    return Minibatch(
        obs=obs.reshape(num_minibatches, minibatch_size, obs_dim),
        actions=actions.reshape(num_minibatches, minibatch_size),
        log_probs=log_probs.reshape(num_minibatches, minibatch_size),
        advantages=advantages.reshape(num_minibatches, minibatch_size),
        returns=returns.reshape(num_minibatches, minibatch_size),
        values=values.reshape(num_minibatches, minibatch_size),
    )


def loss_fn(
    model_params: ModelParams,
    model: ActorCritic,
    minibatch: Minibatch,
    clip_coef: float,
    clip_vloss: bool,
    ent_coef: float,
    vf_coef: float,
    rng_key: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]]:
    """Compute PPO loss for a minibatch.
    
    Args:
        model_params: Model parameters.
        model: ActorCritic module.
        minibatch: Minibatch of training data.
        clip_coef: PPO clipping coefficient.
        clip_vloss: Whether to clip value loss.
        ent_coef: Entropy coefficient.
        vf_coef: Value function coefficient.
        rng_key: PRNG key (unused, but needed for API compatibility).
    
    Returns:
        Tuple of (total_loss, (policy_loss, value_loss, entropy_loss, approx_kl, clip_frac)).
    """
    # Forward pass: get new log_prob, entropy, value for the stored actions
    result = model.apply(
        model_params,
        minibatch.obs,
        rng_key,  # not used when action is provided
        minibatch.actions,
        method=model.get_action_and_value,
    )
    _, new_log_prob, entropy, new_value = cast(ModelActionValueResult, result)
    
    # Compute ratio
    log_ratio = new_log_prob - minibatch.log_probs
    ratio = jnp.exp(log_ratio)
    
    # Debug metrics: approx KL and clip fraction
    approx_kl = ((ratio - 1) - log_ratio).mean()
    clip_frac = (jnp.abs(ratio - 1.0) > clip_coef).astype(jnp.float32).mean()
    
    # Advantage normalization (per minibatch)
    advantages = minibatch.advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Clipped surrogate objective (Detail 8)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * jnp.clip(ratio, 1 - clip_coef, 1 + clip_coef)
    pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()
    
    # Value function loss (Detail 9: optionally clipped)
    if clip_vloss:
        v_loss_unclipped = (new_value - minibatch.returns) ** 2
        v_clipped = minibatch.values + jnp.clip(
            new_value - minibatch.values, -clip_coef, clip_coef
        )
        v_loss_clipped = (v_clipped - minibatch.returns) ** 2
        v_loss = 0.5 * jnp.maximum(v_loss_unclipped, v_loss_clipped).mean()
    else:
        v_loss = 0.5 * ((new_value - minibatch.returns) ** 2).mean()
    
    # Entropy loss (Detail 10)
    entropy_loss = entropy.mean()
    
    # Total loss
    loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss
    
    return loss, (pg_loss, v_loss, entropy_loss, approx_kl, clip_frac)


def update(
    config: PPOConfig,
    model: ActorCritic,
    optimizer: optax.GradientTransformation,
    train_state: TrainState,
    rollout_with_gae: RolloutWithGAE,
    rng_key: jax.Array,
) -> tuple[TrainState, dict[str, jax.Array]]:
    """Perform full PPO update: multiple epochs over minibatches.
    
    Structure: outer scan over epochs, inner scan over minibatches.
    Each minibatch update computes gradients via loss_fn and applies them.
    
    Args:
        config: PPO configuration.
        model: ActorCritic module.
        optimizer: Optax optimizer.
        train_state: Current training state.
        rollout_with_gae: Rollout data with advantages and returns.
        rng_key: PRNG key for shuffling.
    
    Returns:
        Tuple of (updated TrainState, update_metrics dict for logging).
    """
    
    def minibatch_step(carry: UpdateCarry, minibatch: Minibatch) -> tuple[UpdateCarry, None]:
        """Single gradient update on one minibatch."""
        rng_key_, rng_loss = jax.random.split(carry.rng_key)
        
        # Compute loss and gradients
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            carry.train_state.model_params,
            model,
            minibatch,
            config.clip_coef,
            config.clip_vloss,
            config.ent_coef,
            config.vf_coef,
            rng_loss,
        )
        pg_loss, v_loss, entropy_loss, approx_kl, clip_frac = aux
        
        # Apply gradients
        updates, new_opt_state = optimizer.update(
            grads, carry.train_state.opt_state, carry.train_state.model_params
        )
        new_params = cast(ModelParams, optax.apply_updates(carry.train_state.model_params, updates))
        
        # Update metrics accumulator
        accum = carry.metrics_accum
        new_metrics_accum = UpdateMetricsAccum(
            sum_total_loss=accum.sum_total_loss + loss,
            sum_policy_loss=accum.sum_policy_loss + pg_loss,
            sum_value_loss=accum.sum_value_loss + v_loss,
            sum_entropy_loss=accum.sum_entropy_loss + entropy_loss,
            sum_approx_kl=accum.sum_approx_kl + approx_kl,
            sum_clip_frac=accum.sum_clip_frac + clip_frac,
            n_updates=accum.n_updates + 1,
        )
        
        new_carry = UpdateCarry(
            train_state=TrainState(
                model_params=new_params,
                opt_state=new_opt_state,
                step=carry.train_state.step + 1,
            ),
            metrics_accum=new_metrics_accum,
            rng_key=rng_key_,
        )
        return new_carry, None
    
    def epoch_step(carry: UpdateCarry, _: None) -> tuple[UpdateCarry, None]:
        """One epoch: shuffle data and scan over all minibatches."""
        rng_key_, rng_shuffle = jax.random.split(carry.rng_key)
        
        # Prepare minibatches with fresh shuffle
        minibatches = get_minibatches(rollout_with_gae, config.num_minibatches, rng_shuffle)
        
        # Scan over minibatches
        new_carry, _ = jax.lax.scan(
            minibatch_step,
            init=UpdateCarry(
                train_state=carry.train_state,
                metrics_accum=carry.metrics_accum,
                rng_key=rng_key_,
            ),
            xs=minibatches,
        )
        return new_carry, None
    
    # Initialize carry with fresh metrics accumulator
    init_carry = UpdateCarry(
        train_state=train_state,
        metrics_accum=UpdateMetricsAccum(),
        rng_key=rng_key,
    )
    
    # Scan over epochs
    final_carry, _ = jax.lax.scan(
        epoch_step,
        init=init_carry,
        xs=None,
        length=config.update_epochs,
    )
    
    # Convert accumulated sums to metrics dict for logging
    update_metrics = final_carry.metrics_accum.to_metrics_dict()
    
    # Add explained variance (computed from rollout, not accumulated)
    flat_values = rollout_with_gae.values.reshape(-1)
    flat_returns = rollout_with_gae.returns.reshape(-1)
    var_returns = jnp.var(flat_returns)
    update_metrics["explained_variance"] = jnp.where(
        var_returns == 0,
        jnp.nan,
        1 - jnp.var(flat_returns - flat_values) / var_returns,
    )
    
    return final_carry.train_state, update_metrics
