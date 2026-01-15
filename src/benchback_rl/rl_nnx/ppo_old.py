"""Proximal Policy Optimization (PPO) implementation using Flax NNX.

Implements the 13 core implementation details from:
https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/

This is a stateful implementation using NNX modules, following the PyTorch
structure closely while supporting three different JIT compilation modes:
1. nnx.jit - NNX handles split/merge automatically
2. cached_partial - Caches graph traversals for reduced overhead
3. manual - Explicit split/merge with jax.jit for comparison
"""

from typing import Any, Literal
import math
import time

import jax
import jax.numpy as jnp
import optax
import wandb
from flax import nnx, struct
from tqdm import tqdm

from benchback_rl.rl_common import PPOConfig
from benchback_rl.rl_nnx.models import ActorCritic

# Type aliases for gymnax environments
Env = Any
EnvParams = Any
EnvState = Any

JitMode = Literal["nnx", "cached_partial", "manual"]


@struct.dataclass
class RolloutData:
    """Rollout data with computed GAE advantages and returns.
    
    All arrays have shape (num_steps, num_envs, ...).
    """
    obs: jax.Array          # (num_steps, num_envs, obs_dim)
    actions: jax.Array      # (num_steps, num_envs)
    log_probs: jax.Array    # (num_steps, num_envs)
    values: jax.Array       # (num_steps, num_envs)
    advantages: jax.Array   # (num_steps, num_envs)
    returns: jax.Array      # (num_steps, num_envs)


@struct.dataclass
class EpisodeMetrics:
    """Episode statistics collected during rollouts."""
    n_completed: jax.Array
    sum_rewards: jax.Array
    sum_lengths: jax.Array
    sum_reward_per_step: jax.Array
    min_rewards: jax.Array
    max_rewards: jax.Array
    min_length: jax.Array
    max_length: jax.Array

    @classmethod
    def empty(cls) -> "EpisodeMetrics":
        """Create empty accumulators."""
        return cls(
            n_completed=jnp.array(0, dtype=jnp.int32),
            sum_rewards=jnp.array(0.0),
            sum_lengths=jnp.array(0.0),
            sum_reward_per_step=jnp.array(0.0),
            min_rewards=jnp.array(jnp.inf),
            max_rewards=jnp.array(-jnp.inf),
            min_length=jnp.array(jnp.inf),
            max_length=jnp.array(-jnp.inf),
        )

    def to_dict(self) -> dict[str, jax.Array]:
        """Convert to metrics dict with NaN for empty."""
        n = jnp.maximum(self.n_completed, 1)
        any_completed = self.n_completed > 0
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


class PPO(nnx.Module):
    """Proximal Policy Optimization algorithm using Flax NNX.
    
    Stateful class where rollout and update methods modify instance attributes.
    Supports three JIT modes for benchmarking different compilation strategies.
    """

    def __init__(
        self,
        env: Env,
        env_params: EnvParams,
        model: ActorCritic,
        config: PPOConfig,
        jit_mode: JitMode = "nnx",
    ) -> None:
        """Initialize PPO trainer.
        
        Args:
            env: Gymnax environment (single, not vmapped).
            env_params: Gymnax environment parameters.
            model: ActorCritic NNX module.
            config: PPO configuration.
            jit_mode: JIT compilation strategy ("nnx", "cached_partial", or "manual").
        """
        # Verify config matches expected framework
        if config.framework != "nnx":
            raise ValueError(f"Expected framework='nnx', got '{config.framework}'")

        # ===== 1. ALL nnx.Variable state first =====
        
        # Model and optimizer (nnx.Module contains Variables)
        self.model = model
        
        # Create optimizer with linear LR schedule annealing to 0
        schedule = optax.linear_schedule(
            init_value=config.learning_rate,
            end_value=0.0,
            transition_steps=config.total_optimizer_steps,
        )
        tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(
                learning_rate=schedule,
                eps=config.adam_eps,
                b1=config.adam_betas[0],
                b2=config.adam_betas[1],
            ),
        )
        self.optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
        
        # RNG state
        seed = config.seed if config.seed is not None else int(time.time_ns())
        self.rngs = nnx.Rngs(seed=seed)
        
        # Environment state (wrapped in nnx.Variable for tracking)
        # Will be initialized properly in reset()
        obs_shape = (config.num_envs, env.observation_space(env_params).shape[0])
        self.env_obs = nnx.Variable(jnp.zeros(obs_shape))
        self.env_state = nnx.Variable(None)  # Will hold vmapped env state
        
        # Episode tracking (per-env accumulators)
        self.episode_rewards = nnx.Variable(jnp.zeros(config.num_envs))
        self.episode_lengths = nnx.Variable(jnp.zeros(config.num_envs, dtype=jnp.int32))

        # ===== 2. Non-Variable config (static, not in state) =====
        self.config = config
        self.env = env
        self.env_params = env_params
        self._jit_mode = jit_mode
        
        # Create vmapped environment functions
        self._env_reset = jax.vmap(env.reset, in_axes=(0, None))
        self._env_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

        # ===== 3. JIT setup LAST (after all Variables exist) =====
        if jit_mode == "nnx":
            # Use unbound method + partial to create jitted methods
            # (bound methods like self.method are not supported by nnx.jit)
            _collect_rollout_jit = nnx.jit(PPO._collect_rollout_impl)
            _update_jit = nnx.jit(PPO._update_impl)
            
            self._collect_rollout = lambda: _collect_rollout_jit(self)
            self._update = lambda rollout: _update_jit(self, rollout)
            
        elif jit_mode == "cached_partial":
            self._collect_rollout = nnx.cached_partial(
                nnx.jit(PPO._collect_rollout_impl), self
            )
            self._update = nnx.cached_partial(
                nnx.jit(PPO._update_impl), self
            )
            
        elif jit_mode == "manual":
            # Capture graphdef now that all Variables are defined
            graphdef, _ = nnx.split(self)
            
            @jax.jit
            def collect_rollout_pure(state):
                ppo = nnx.merge(graphdef, state)
                result = ppo._collect_rollout_impl()
                return nnx.state(ppo), result
            
            @jax.jit
            def update_pure(state, rollout):
                ppo = nnx.merge(graphdef, state)
                result = ppo._update_impl(rollout)
                return nnx.state(ppo), result
            
            def collect_rollout_wrapper():
                state = nnx.state(self)
                new_state, result = collect_rollout_pure(state)
                nnx.update(self, new_state)
                return result
            
            def update_wrapper(rollout):
                state = nnx.state(self)
                new_state, result = update_pure(state, rollout)
                nnx.update(self, new_state)
                return result
            
            self._collect_rollout = collect_rollout_wrapper
            self._update = update_wrapper
        else:
            raise ValueError(f"Unknown jit_mode: {jit_mode}")

    def _time(self) -> float:
        """Get current time, optionally syncing JAX first for accurate timing."""
        if self.config.sync_for_timing:
            # Block until all JAX computations are complete
            jax.block_until_ready(nnx.state(self.model))
        return time.perf_counter()

    def _log_hparams(self) -> None:
        """Log hyperparameters to WandB if a run is active."""
        if wandb.run is None:
            return
        wandb.config.update(self.config.to_dict())

    def _log_metrics(self, metrics: dict[str, float], iteration: int) -> None:
        """Log metrics to WandB if a run is active."""
        if wandb.run is None:
            return
        wandb.log({"iteration": iteration, **metrics}, step=iteration)

    def _log_summary(self, summary: dict[str, Any]) -> None:
        """Log final training summary to WandB if a run is active."""
        if wandb.run is None:
            return
        for key, value in summary.items():
            wandb.run.summary[key] = value

    def _collect_rollout_impl(self) -> tuple[RolloutData, dict[str, jax.Array]]:
        """Collect a rollout of experience and compute GAE.
        
        Modifies: self.env_obs, self.env_state, self.episode_rewards,
                  self.episode_lengths, self.rngs
        
        Returns:
            Tuple of (RolloutData, episode_metrics dict).
        """
        config = self.config
        
        def step(carry, _):
            """Single step of rollout collection."""
            obs, env_state, episode_rewards, episode_lengths, metrics_accum, rng_key = carry
            
            rng_key, rng_action, rng_step = jax.random.split(rng_key, 3)
            
            # Get action and value from model
            action, log_prob, _, value = self.model.get_action_and_value(obs, rng_action)
            
            # Step environment
            rng_keys_step = jax.random.split(rng_step, config.num_envs)
            next_obs, next_env_state, reward, done, _ = self._env_step(
                rng_keys_step, env_state, action, self.env_params
            )
            
            # Update episode trackers
            episode_rewards = episode_rewards + reward
            episode_lengths = episode_lengths + 1
            
            # Update metrics accumulators for completed episodes
            new_metrics = EpisodeMetrics(
                n_completed=metrics_accum.n_completed + done.sum().astype(jnp.int32),
                sum_rewards=metrics_accum.sum_rewards + (episode_rewards * done).sum(),
                sum_lengths=metrics_accum.sum_lengths + (episode_lengths * done).sum(),
                sum_reward_per_step=metrics_accum.sum_reward_per_step + (episode_rewards / episode_lengths * done).sum(),
                min_rewards=jnp.minimum(metrics_accum.min_rewards,
                                        jnp.min(jnp.asarray(jnp.where(done, episode_rewards, jnp.inf)))),
                max_rewards=jnp.maximum(metrics_accum.max_rewards,
                                        jnp.max(jnp.asarray(jnp.where(done, episode_rewards, -jnp.inf)))),
                min_length=jnp.minimum(metrics_accum.min_length,
                                       jnp.min(jnp.asarray(jnp.where(done, episode_lengths, jnp.inf)))),
                max_length=jnp.maximum(metrics_accum.max_length,
                                       jnp.max(jnp.asarray(jnp.where(done, episode_lengths, -jnp.inf)))),
            )
            
            # Reset episode trackers on done
            episode_rewards = jnp.asarray(jnp.where(done, 0.0, episode_rewards))
            episode_lengths = jnp.asarray(jnp.where(done, 0, episode_lengths))
            
            # Carry and output
            new_carry = (next_obs, next_env_state, episode_rewards, episode_lengths, new_metrics, rng_key)
            transition = (obs, action, log_prob, value, reward, done)
            
            return new_carry, transition
        
        # Initialize - get key from rngs for this rollout
        rollout_key = self.rngs.rollout()
        
        init_carry = (
            self.env_obs.value,
            self.env_state.value,
            self.episode_rewards.value,
            self.episode_lengths.value,
            EpisodeMetrics.empty(),
            rollout_key,
        )
        
        # Run rollout
        final_carry, transitions = jax.lax.scan(
            step,
            init_carry,
            xs=None,
            length=config.num_steps,
        )
        
        final_obs, final_env_state, final_ep_rewards, final_ep_lengths, final_metrics, _ = final_carry
        obs, actions, log_probs, values, rewards, dones = transitions
        
        # Update state
        self.env_obs.value = final_obs
        self.env_state.value = final_env_state
        self.episode_rewards.value = final_ep_rewards
        self.episode_lengths.value = final_ep_lengths
        
        # Compute bootstrap value
        final_value = self.model.get_value(final_obs)
        
        # Compute GAE
        advantages, returns = self._compute_gae(values, rewards, dones, final_value)
        
        rollout = RolloutData(
            obs=obs,
            actions=actions,
            log_probs=log_probs,
            values=values,
            advantages=advantages,
            returns=returns,
        )
        
        return rollout, final_metrics.to_dict()

    def _compute_gae(
        self,
        values: jax.Array,
        rewards: jax.Array,
        dones: jax.Array,
        final_value: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Compute Generalized Advantage Estimation (GAE).
        
        Args:
            values: Value estimates, shape (num_steps, num_envs).
            rewards: Rewards, shape (num_steps, num_envs).
            dones: Done flags, shape (num_steps, num_envs).
            final_value: Value of final observation, shape (num_envs,).
        
        Returns:
            Tuple of (advantages, returns), each shape (num_steps, num_envs).
        """
        config = self.config
        
        def gae_step(carry, xs):
            next_gae, next_value = carry
            value, reward, done = xs
            
            not_done = 1.0 - done
            delta = reward + config.gamma * next_value * not_done - value
            gae = delta + config.gamma * config.gae_lambda * not_done * next_gae
            
            return (gae, value), gae
        
        num_envs = final_value.shape[0]
        init_carry = (jnp.zeros(num_envs), final_value)
        
        _, advantages = jax.lax.scan(
            gae_step,
            init_carry,
            (values, rewards, dones),
            reverse=True,
        )
        
        returns = advantages + values
        return advantages, returns

    def _update_impl(self, rollout: RolloutData) -> dict[str, jax.Array]:
        """Perform PPO update on collected rollout.
        
        Modifies: self.model (via self.optimizer), self.rngs
        
        Args:
            rollout: Collected rollout data with advantages.
        
        Returns:
            Dict with update metrics.
        """
        config = self.config
        
        # Flatten rollout data
        batch_size = config.num_steps * config.num_envs
        obs_dim = rollout.obs.shape[-1]
        
        flat_obs = rollout.obs.reshape(batch_size, obs_dim)
        flat_actions = rollout.actions.reshape(batch_size)
        flat_log_probs = rollout.log_probs.reshape(batch_size)
        flat_advantages = rollout.advantages.reshape(batch_size)
        flat_returns = rollout.returns.reshape(batch_size)
        flat_values = rollout.values.reshape(batch_size)
        
        def loss_fn(model, mb_obs, mb_actions, mb_log_probs, mb_advantages, mb_returns, mb_values, rng_key):
            """PPO loss for a minibatch."""
            # Forward pass
            _, new_log_prob, entropy, new_value = model.get_action_and_value(
                mb_obs, rng_key, mb_actions
            )
            
            # Ratio
            log_ratio = new_log_prob - mb_log_probs
            ratio = jnp.exp(log_ratio)
            
            # Debug metrics
            approx_kl = ((ratio - 1) - log_ratio).mean()
            clip_frac = (jnp.abs(ratio - 1.0) > config.clip_coef).astype(jnp.float32).mean()
            
            # Advantage normalization (per minibatch)
            advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
            
            # Clipped surrogate objective
            pg_loss1 = -advantages * ratio
            pg_loss2 = -advantages * jnp.clip(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
            pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()
            
            # Value loss
            if config.clip_vloss:
                v_loss_unclipped = (new_value - mb_returns) ** 2
                v_clipped = mb_values + jnp.clip(
                    new_value - mb_values, -config.clip_coef, config.clip_coef
                )
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss = 0.5 * jnp.maximum(v_loss_unclipped, v_loss_clipped).mean()
            else:
                v_loss = 0.5 * ((new_value - mb_returns) ** 2).mean()
            
            # Entropy loss
            entropy_loss = entropy.mean()
            
            # Total loss
            loss = pg_loss - config.ent_coef * entropy_loss + config.vf_coef * v_loss
            
            return loss, (pg_loss, v_loss, entropy_loss, approx_kl, clip_frac)
        
        # Metrics accumulators
        sum_total_loss = 0.0
        sum_policy_loss = 0.0
        sum_value_loss = 0.0
        sum_entropy_loss = 0.0
        sum_approx_kl = 0.0
        sum_clip_frac = 0.0
        n_updates = 0
        
        # Python loops over epochs and minibatches (unrolled at trace time)
        # This avoids the NNX transform nesting issues with jax.lax.scan
        for _epoch in range(config.update_epochs):
            # Shuffle indices for this epoch
            rng_shuffle = self.rngs.shuffle()
            indices = jax.random.permutation(rng_shuffle, batch_size)
            
            # Shuffle all data
            shuf_obs = flat_obs[indices]
            shuf_actions = flat_actions[indices]
            shuf_log_probs = flat_log_probs[indices]
            shuf_advantages = flat_advantages[indices]
            shuf_returns = flat_returns[indices]
            shuf_values = flat_values[indices]
            
            # Reshape into minibatches
            mb_size = config.minibatch_size
            n_mb = config.num_minibatches
            
            mb_obs = shuf_obs.reshape(n_mb, mb_size, obs_dim)
            mb_actions = shuf_actions.reshape(n_mb, mb_size)
            mb_log_probs = shuf_log_probs.reshape(n_mb, mb_size)
            mb_advantages = shuf_advantages.reshape(n_mb, mb_size)
            mb_returns = shuf_returns.reshape(n_mb, mb_size)
            mb_values = shuf_values.reshape(n_mb, mb_size)
            
            for mb_idx in range(n_mb):
                obs_mb = mb_obs[mb_idx]
                act_mb = mb_actions[mb_idx]
                lp_mb = mb_log_probs[mb_idx]
                adv_mb = mb_advantages[mb_idx]
                ret_mb = mb_returns[mb_idx]
                val_mb = mb_values[mb_idx]
                
                rng_loss = self.rngs.loss()
                
                # Compute loss and gradients using nnx.value_and_grad
                (loss, aux), grads = nnx.value_and_grad(
                    loss_fn, argnums=nnx.DiffState(0, nnx.Param), has_aux=True
                )(self.model, obs_mb, act_mb, lp_mb, adv_mb, ret_mb, val_mb, rng_loss)
                pg_loss, v_loss, ent_loss, approx_kl, clip_frac = aux
                
                # Apply gradients
                self.optimizer.update(self.model, grads)
                
                # Accumulate metrics
                sum_total_loss = sum_total_loss + loss
                sum_policy_loss = sum_policy_loss + pg_loss
                sum_value_loss = sum_value_loss + v_loss
                sum_entropy_loss = sum_entropy_loss + ent_loss
                sum_approx_kl = sum_approx_kl + approx_kl
                sum_clip_frac = sum_clip_frac + clip_frac
                n_updates = n_updates + 1
        
        n = jnp.array(n_updates, dtype=jnp.float32)
        
        # Explained variance
        var_returns = jnp.var(flat_returns)
        explained_var = jnp.where(
            var_returns == 0,
            jnp.nan,
            1 - jnp.var(flat_returns - flat_values) / var_returns,
        )
        
        return {
            "total_loss": sum_total_loss / n,
            "policy_loss": sum_policy_loss / n,
            "value_loss": sum_value_loss / n,
            "entropy_loss": sum_entropy_loss / n,
            "approx_kl": sum_approx_kl / n,
            "clip_frac": sum_clip_frac / n,
            "explained_variance": explained_var,
        }

    def reset(self) -> None:
        """Reset environment and prepare for fresh training."""
        reset_key = self.rngs.reset()
        reset_keys = jax.random.split(reset_key, self.config.num_envs)
        
        obs, env_state = self._env_reset(reset_keys, self.env_params)
        
        self.env_obs.value = obs
        self.env_state.value = env_state
        self.episode_rewards.value = jnp.zeros(self.config.num_envs)
        self.episode_lengths.value = jnp.zeros(self.config.num_envs, dtype=jnp.int32)

    def train_step(self) -> dict[str, float]:
        """Perform one complete training step (rollout + update).
        
        Returns:
            Dict with combined metrics from rollout and update, plus timing info.
        """
        time_rollout_start = self._time()
        
        rollout, episode_metrics = self._collect_rollout()
        
        time_rollout_end = self._time()
        
        update_metrics = self._update(rollout)
        
        time_update_end = self._time()
        
        # Combine and convert to Python floats
        metrics: dict[str, float] = {}
        for key, value in episode_metrics.items():
            metrics[key] = float(value)
        for key, value in update_metrics.items():
            metrics[key] = float(value)
        
        metrics["duration_rollout"] = time_rollout_end - time_rollout_start
        metrics["duration_update"] = time_update_end - time_rollout_end
        
        return metrics

    def train_from_scratch(self) -> None:
        """Run the full PPO training loop."""
        time_start = self._time()
        time_step_end = time_start

        duration_first_step = 0.0
        duration_first_overhead = 0.0
        duration_second_plus_step_sum = 0.0
        duration_second_plus_overhead_sum = 0.0

        self.reset()
        self._log_hparams()

        pbar = tqdm(range(self.config.num_iterations), desc="Training")
        for iteration in pbar:
            # Timing
            time_iteration_start = self._time()
            duration_overhead = time_iteration_start - time_step_end
            if iteration == 0:
                duration_first_overhead = duration_overhead
            else:
                duration_second_plus_overhead_sum += duration_overhead

            # TRAINING STEP
            metrics = self.train_step()

            # Timing
            time_step_end = self._time()
            duration_step = time_step_end - time_iteration_start
            if iteration == 0:
                duration_first_step = duration_step
            else:
                duration_second_plus_step_sum += duration_step

            metrics["duration_step"] = duration_step
            metrics["duration_step_overhead"] = duration_step - metrics["duration_rollout"] - metrics["duration_update"]
            metrics["duration_overhead"] = duration_overhead
            metrics["time_elapsed"] = time_step_end - time_start

            self._log_metrics(metrics, iteration)

            # Progress bar
            avg_reward = metrics.get("avg_episode_reward", float("nan"))
            approx_kl = metrics.get("approx_kl", float("nan"))
            reward_str = f"{avg_reward:.1f}" if not math.isnan(avg_reward) else "N/A"
            kl_str = f"{approx_kl:.4f}" if not math.isnan(approx_kl) else "N/A"
            pbar.set_postfix({"reward": reward_str, "kl": kl_str})

        pbar.close()

        # Final timing summary
        time_end = self._time()
        duration_total = time_end - time_start
        duration_average_second_plus_overhead = duration_second_plus_overhead_sum / max(1, self.config.num_iterations - 1)
        duration_average_second_plus_step = duration_second_plus_step_sum / max(1, self.config.num_iterations - 1)
        
        self._log_summary({
            "duration_total": duration_total,
            "duration_first_step": duration_first_step,
            "duration_first_overhead": duration_first_overhead,
            "duration_average_second_plus_step": duration_average_second_plus_step,
            "duration_average_second_plus_overhead": duration_average_second_plus_overhead,
        })

        print(f"\nTraining completed in {duration_total/60:.1f} minutes")
        print(f"First iteration time: {duration_first_step+duration_first_overhead:.3f}s")
        print(f"Average time per iteration (excluding first): "
              f"{duration_average_second_plus_step+duration_average_second_plus_overhead:.3f}s")
        print(f"Final average reward: {reward_str}")
