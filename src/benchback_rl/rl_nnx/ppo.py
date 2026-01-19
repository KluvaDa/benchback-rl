"""Proximal Policy Optimization (PPO) implementation using Flax NNX.

Implements the 13 core implementation details from:
https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/

Design choices:
- done[t] indicates episode ended AFTER taking action[t] (not before observing obs[t])
- All terminations (true terminal states and time-limit truncations) are treated the same
- obs[0:num_steps+1] with the extra slot for the bootstrap observation
- rngs.minibatch() used for shuffling minibatch indices each epoch
"""
import math
import time
from typing import Any

import jax
import jax.numpy as jnp
import optax
import wandb
from flax import nnx
from tqdm import tqdm

from benchback_rl.rl_common.config import PPOConfig
from benchback_rl.rl_nnx.env import NnxVecEnv
from benchback_rl.rl_nnx.models import ActorCritic


class PPOVariable(nnx.Variable):
    pass


class PPO(nnx.Module):
    """Proximal Policy Optimization (PPO) using Flax NNX.
    Stateful class where methods modify self.PPOVariable objects using .value
    """
    def __init__(
        self,
        config: PPOConfig,
        env: NnxVecEnv,
        model: ActorCritic,
        rngs: nnx.Rngs,
        start_time: float | None = None,
    ) -> None:
        """
        Args:
            config: PPO hyperparameter configuration.
            env: NnxVecEnv instance.
            model: ActorCritic model instance.
            rngs: nnx.Rngs object used with rngs.minibatch()
            start_time: Time at beginning of main for initial overhead calculation.
        """
        if config.framework != "nnx":
            raise ValueError(f"Expected framework='nnx', got '{config.framework}'")
        
        # store arguments
        self.config = config  # static
        self.env = env  # stateful nnx.Module
        self.model = model # stateful nnx.Module
        self.rngs = rngs  # stateful nnx.Rngs
        self.start_time = start_time

        # Optimizer with linear LR decay
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
        self.optimizer = nnx.Optimizer(self.model, tx, wrt=nnx.Param)

        # Rollout storage (obs and values have one extra slot for bootstrap)
        self._obs = PPOVariable(jnp.empty((config.num_steps+1, env.num_envs, env.obs_dim)))
        self._values = PPOVariable(jnp.empty((config.num_steps+1, env.num_envs)))
        self._actions = PPOVariable(jnp.empty((config.num_steps, env.num_envs), dtype=jnp.int32))
        self._log_probs = PPOVariable(jnp.empty((config.num_steps, env.num_envs)))
        self._rewards = PPOVariable(jnp.empty((config.num_steps, env.num_envs)))
        self._dones = PPOVariable(jnp.empty((config.num_steps, env.num_envs), dtype=bool))
        # GAE storage
        self._advantages = PPOVariable(jnp.empty((config.num_steps, env.num_envs)))
        self._returns = PPOVariable(jnp.empty((config.num_steps, env.num_envs)))
        self._next_gae = PPOVariable(jnp.empty((env.num_envs,)))

        # Episode statistics (per-env, reset on done)
        self._episode_rewards = PPOVariable(jnp.empty((env.num_envs,)))
        self._episode_lengths = PPOVariable(jnp.empty((env.num_envs,)))

        # Rollout metrics accumulators (reset at start of each rollout)
        self._n_completed = PPOVariable(jnp.empty(1, dtype=jnp.int32))
        self._sum_rewards = PPOVariable(jnp.empty(1))
        self._sum_lengths = PPOVariable(jnp.empty(1))
        self._sum_reward_per_step = PPOVariable(jnp.empty(1))
        self._min_rewards = PPOVariable(jnp.empty(1))
        self._max_rewards = PPOVariable(jnp.empty(1))
        self._min_length = PPOVariable(jnp.empty(1))
        self._max_length = PPOVariable(jnp.empty(1))

        # Update metrics accumulators (reset at start of each update)
        self._total_policy_loss = PPOVariable(jnp.empty(1))
        self._total_value_loss = PPOVariable(jnp.empty(1))
        self._total_entropy_loss = PPOVariable(jnp.empty(1))
        self._total_loss = PPOVariable(jnp.empty(1))
        self._total_approx_kl = PPOVariable(jnp.empty(1))
        self._total_clip_frac = PPOVariable(jnp.empty(1))

        # Update phase storage (flattened tensors and shuffled indices)
        self._flat_obs = PPOVariable(jnp.empty((config.batch_size, env.obs_dim)))
        self._flat_actions = PPOVariable(jnp.empty((config.batch_size,), dtype=jnp.int32))
        self._flat_log_probs = PPOVariable(jnp.empty((config.batch_size,)))
        self._flat_advantages = PPOVariable(jnp.empty((config.batch_size,)))
        self._flat_returns = PPOVariable(jnp.empty((config.batch_size,)))
        self._flat_values = PPOVariable(jnp.empty((config.batch_size,)))
        self._shuffled_indices = PPOVariable(jnp.empty((config.batch_size,), dtype=jnp.int32))

        # JIT compilation based on config.compile
        if config.compile == "none":
            self._collect_rollout = lambda: PPO._collect_rollout_impl(self)
            self._compute_gae = lambda: PPO._compute_gae_impl(self)
            self._update = lambda: PPO._update_impl(self)
        elif config.compile == "nnx.jit":
            _collect_rollout_jit = nnx.jit(PPO._collect_rollout_impl)
            _compute_gae_jit = nnx.jit(PPO._compute_gae_impl)
            _update_jit = nnx.jit(PPO._update_impl)
            self._collect_rollout = lambda: _collect_rollout_jit(self)
            self._compute_gae = lambda: _compute_gae_jit(self)
            self._update = lambda: _update_jit(self)
        elif config.compile == "nnx.cached_partial":
            self._collect_rollout = nnx.cached_partial(nnx.jit(PPO._collect_rollout_impl), self)
            self._compute_gae = nnx.cached_partial(nnx.jit(PPO._compute_gae_impl), self)
            self._update = nnx.cached_partial(nnx.jit(PPO._update_impl), self)
        else:
            raise ValueError(f"Unsupported compile mode for nnx: {config.compile}")
        
    def _time(self, block_until_ready_object: Any|None = None) -> float:
        """Get current time, optionally syncing JAX first for accurate timing."""
        if self.config.sync_for_timing and block_until_ready_object is not None:
            jax.block_until_ready(block_until_ready_object)
        return time.perf_counter()

    def _log_hparams(self) -> None:
        """Log hyperparameters to wandb if enabled."""
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
    
    def _collect_rollout_impl(self) -> dict[str, jax.Array]:
        """Collect rollout data and store in instance variables.
        
        Uses nnx.scan for stateful rollout collection.
        Directly mutates instance buffers (_obs, _actions, etc.) inside the loop.
        
        Returns:
            Episode metrics for completed episodes during this rollout.
        """
        # Reset metrics accumulator for this rollout
        self._n_completed.value = jnp.array(0, dtype=jnp.int32)
        self._sum_rewards.value = jnp.array(0.0)
        self._sum_lengths.value = jnp.array(0.0)
        self._sum_reward_per_step.value = jnp.array(0.0)
        self._min_rewards.value = jnp.array(jnp.inf)
        self._max_rewards.value = jnp.array(-jnp.inf)
        self._min_length.value = jnp.array(jnp.inf)
        self._max_length.value = jnp.array(-jnp.inf)
        
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        def step_fn(ppo: "PPO", i: jax.Array) -> "PPO":
            """Single rollout step with stateful buffer updates."""
            obs = ppo._obs.value[i]
            action, log_prob, _, value = ppo.model.get_action_and_value(obs)
            next_obs, reward, done, _ = ppo.env.step(action)
            
            # Store transition
            ppo._obs.value = ppo._obs.value.at[i + 1].set(next_obs)
            ppo._actions.value = ppo._actions.value.at[i].set(action)
            ppo._log_probs.value = ppo._log_probs.value.at[i].set(log_prob)
            ppo._values.value = ppo._values.value.at[i].set(value)
            ppo._rewards.value = ppo._rewards.value.at[i].set(reward)
            ppo._dones.value = ppo._dones.value.at[i].set(done)
            
            # Update episode trackers
            ppo._episode_rewards.value = ppo._episode_rewards.value + reward
            ppo._episode_lengths.value = ppo._episode_lengths.value + 1
            
            # Update metrics accumulators for completed episodes
            ppo._n_completed.value = ppo._n_completed.value \
                + done.sum().astype(jnp.int32)
            ppo._sum_rewards.value = ppo._sum_rewards.value \
                + (ppo._episode_rewards.value * done).sum()
            ppo._sum_lengths.value = ppo._sum_lengths.value \
                + (ppo._episode_lengths.value * done).sum()
            ppo._sum_reward_per_step.value = ppo._sum_reward_per_step.value \
                + (ppo._episode_rewards.value / ppo._episode_lengths.value * done).sum()
            ppo._min_rewards.value = jnp.minimum(ppo._min_rewards.value,
                jnp.min(jnp.asarray(jnp.where(done, ppo._episode_rewards.value, jnp.inf))))
            ppo._max_rewards.value = jnp.maximum(ppo._max_rewards.value,
                jnp.max(jnp.asarray(jnp.where(done, ppo._episode_rewards.value, -jnp.inf))))
            ppo._min_length.value = jnp.minimum(ppo._min_length.value,
                jnp.min(jnp.asarray(jnp.where(done, ppo._episode_lengths.value, jnp.inf))))
            ppo._max_length.value = jnp.maximum(ppo._max_length.value,
                jnp.max(jnp.asarray(jnp.where(done, ppo._episode_lengths.value, -jnp.inf))))

            # Reset episode trackers on done
            ppo._episode_rewards.value = jnp.where(done, 0.0, ppo._episode_rewards.value)
            ppo._episode_lengths.value = jnp.where(done, 0, ppo._episode_lengths.value)
            
            return ppo
        
        step_fn(self, jnp.arange(self.config.num_steps))
        
        # Bootstrap value for final observation
        final_obs = self._obs.value[self.config.num_steps]
        final_value = self.model.get_value(final_obs)
        self._values.value = self._values.value.at[self.config.num_steps].set(final_value)
        
        # Compute metrics
        n_completed_nonzero = jnp.maximum(self._n_completed.value, 1)
        any_completed = self._n_completed.value > 0
        metrics = {
            "episodes_completed": self._n_completed.value,
            "avg_episode_reward": jnp.where(any_completed, self._sum_rewards.value / n_completed_nonzero, jnp.nan),
            "avg_episode_length": jnp.where(any_completed, self._sum_lengths.value / n_completed_nonzero, jnp.nan),
            "avg_reward_per_step": jnp.where(any_completed, self._sum_reward_per_step.value / n_completed_nonzero, jnp.nan),
            "min_episode_reward": jnp.where(any_completed, self._min_rewards.value, jnp.nan),
            "max_episode_reward": jnp.where(any_completed, self._max_rewards.value, jnp.nan),
            "min_episode_length": jnp.where(any_completed, self._min_length.value, jnp.nan),
            "max_episode_length": jnp.where(any_completed, self._max_length.value, jnp.nan),
        }
        
        return metrics
    
    def _compute_gae_impl(self) -> None:
        """Compute Generalized Advantage Estimation (GAE).
        
        Uses nnx.scan for stateful GAE computation, passing self through the loop.
        """
        # Reset GAE accumulator
        self._next_gae.value = jnp.zeros((self.env.num_envs,))
        
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        def gae_step(ppo: "PPO", t: jax.Array) -> "PPO":
            """Single GAE step (iterating backwards via reversed indices)."""
            # done[t] indicates episode ended after action[t], so next state is from new episode
            not_done = (~ppo._dones.value[t]).astype(jnp.float32)
            delta = (
                ppo._rewards.value[t]
                + ppo.config.gamma * ppo._values.value[t + 1] * not_done
                - ppo._values.value[t]
            )
            ppo._next_gae.value = delta + ppo.config.gamma * ppo.config.gae_lambda * not_done * ppo._next_gae.value
            ppo._advantages.value = ppo._advantages.value.at[t].set(ppo._next_gae.value)
            return ppo
        
        # Scan over reversed indices for backward iteration
        gae_step(self, jnp.arange(self.config.num_steps - 1, -1, -1))
        
        # TD(lambda) returns: advantages + values
        self._returns.value = self._advantages.value + self._values.value[:self.config.num_steps]
    
    @staticmethod
    def _loss_fn(
        model: ActorCritic,
        b_obs: jax.Array,
        b_actions: jax.Array,
        b_log_probs: jax.Array,
        b_advantages: jax.Array,
        b_returns: jax.Array,
        b_values: jax.Array,
        clip_coef: float,
        clip_vloss: bool,
        ent_coef: float,
        vf_coef: float,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Compute PPO loss for a minibatch."""
        # Get current policy outputs
        _, new_log_prob, entropy, new_value = model.get_action_and_value(
            b_obs, b_actions
        )
        
        log_ratio = new_log_prob - b_log_probs
        ratio = jnp.exp(log_ratio)
        
        # Debug variables
        approx_kl = ((ratio - 1) - log_ratio).mean()
        clip_frac = (jnp.abs(ratio - 1.0) > clip_coef).astype(jnp.float32).mean()
        
        # Advantage normalization
        mb_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        
        # Clipped surrogate objective
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * jnp.clip(ratio, 1 - clip_coef, 1 + clip_coef)
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()
        
        # Value function loss (optionally clipped)
        if clip_vloss:
            v_loss_unclipped = (new_value - b_returns) ** 2
            v_clipped = b_values + jnp.clip(
                new_value - b_values, -clip_coef, clip_coef
            )
            v_loss_clipped = (v_clipped - b_returns) ** 2
            v_loss = 0.5 * jnp.maximum(v_loss_unclipped, v_loss_clipped).mean()
        else:
            v_loss = 0.5 * ((new_value - b_returns) ** 2).mean()
        
        # Combined loss
        entropy_loss = entropy.mean()
        loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss
        
        metrics = {
            "policy_loss": pg_loss,
            "value_loss": v_loss,
            "entropy_loss": entropy_loss,
            "total_loss": loss,
            "approx_kl": approx_kl,
            "clip_frac": clip_frac,
        }
        return loss, metrics

    def _update_impl(self) -> dict[str, jax.Array]:
        """Perform PPO update over multiple epochs with minibatch shuffling.
        
        Uses nnx.scan with self passed through for proper NNX state handling.
        """
        # Flatten all tensors for batching and store in PPOVariables
        self._flat_obs.value = self._obs.value[:self.config.num_steps].reshape(self.config.batch_size, -1)
        self._flat_actions.value = self._actions.value.reshape(self.config.batch_size)
        self._flat_log_probs.value = self._log_probs.value.reshape(self.config.batch_size)
        self._flat_advantages.value = self._advantages.value.reshape(self.config.batch_size)
        self._flat_returns.value = self._returns.value.reshape(self.config.batch_size)
        self._flat_values.value = self._values.value[:self.config.num_steps].reshape(self.config.batch_size)
        
        # Reset metrics accumulators
        self._total_policy_loss.value = jnp.array(0.0)
        self._total_value_loss.value = jnp.array(0.0)
        self._total_entropy_loss.value = jnp.array(0.0)
        self._total_loss.value = jnp.array(0.0)
        self._total_approx_kl.value = jnp.array(0.0)
        self._total_clip_frac.value = jnp.array(0.0)
        
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        def minibatch_step(ppo: "PPO", mb_idx: jax.Array) -> "PPO":
            """Single minibatch update."""
            start = mb_idx * ppo.config.minibatch_size
            batch_indices = jax.lax.dynamic_slice(
                ppo._shuffled_indices.value, (start,), (ppo.config.minibatch_size,)
            )
            
            b_obs = ppo._flat_obs.value[batch_indices]
            b_actions = ppo._flat_actions.value[batch_indices]
            b_log_probs = ppo._flat_log_probs.value[batch_indices]
            b_advantages = ppo._flat_advantages.value[batch_indices]
            b_returns = ppo._flat_returns.value[batch_indices]
            b_values = ppo._flat_values.value[batch_indices]
            
            # Compute gradients and update
            grads, batch_metrics = nnx.grad(PPO._loss_fn, has_aux=True)(
                ppo.model, b_obs, b_actions, b_log_probs, b_advantages, b_returns, b_values,
                ppo.config.clip_coef, ppo.config.clip_vloss, ppo.config.ent_coef, ppo.config.vf_coef,
            )
            ppo.optimizer.update(ppo.model, grads)
            
            # Accumulate metrics (stateful)
            ppo._total_policy_loss.value = ppo._total_policy_loss.value + batch_metrics["policy_loss"]
            ppo._total_value_loss.value = ppo._total_value_loss.value + batch_metrics["value_loss"]
            ppo._total_entropy_loss.value = ppo._total_entropy_loss.value + batch_metrics["entropy_loss"]
            ppo._total_loss.value = ppo._total_loss.value + batch_metrics["total_loss"]
            ppo._total_approx_kl.value = ppo._total_approx_kl.value + batch_metrics["approx_kl"]
            ppo._total_clip_frac.value = ppo._total_clip_frac.value + batch_metrics["clip_frac"]
            return ppo
        
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        def epoch_step(ppo: "PPO", epoch: jax.Array) -> "PPO":
            """Single epoch of minibatch updates."""
            ppo._shuffled_indices.value = jax.random.permutation(
                ppo.rngs.minibatch(), ppo.config.batch_size
            )
            minibatch_step(ppo, jnp.arange(ppo.config.num_minibatches))
            return ppo
        
        # Run all epochs
        epoch_step(self, jnp.arange(self.config.update_epochs))
        
        # Average metrics
        num_updates = self.config.update_epochs * self.config.num_minibatches
        avg_metrics = {
            "policy_loss": self._total_policy_loss.value / num_updates,
            "value_loss": self._total_value_loss.value / num_updates,
            "entropy_loss": self._total_entropy_loss.value / num_updates,
            "total_loss": self._total_loss.value / num_updates,
            "approx_kl": self._total_approx_kl.value / num_updates,
            "clip_frac": self._total_clip_frac.value / num_updates,
        }
        
        # Compute explained variance
        explained_var = 1 - (self._flat_returns.value - self._flat_values.value).var() / self._flat_returns.value.var()
        avg_metrics["explained_variance"] = explained_var
        
        return avg_metrics
    
    def reset(self) -> None:
        """Reset the environment and episode trackers."""
        self._obs.value = self._obs.value.at[0].set(self.env.reset())
        self._episode_rewards.value = jnp.zeros((self.env.num_envs,))
        self._episode_lengths.value = jnp.zeros((self.env.num_envs,))

    def train_step(self) -> dict[str, float]:
        """Perform a single training step: rollout, GAE, update.
        Returns:
            Combined metrics from rollout and update phases.
        """
        # Sync on obs buffer to ensure previous iteration's carry-over is complete
        time_rollout_start = self._time(self._obs.value)

        # Collect rollout
        rollout_metrics = self._collect_rollout()

        # compute GAE advantages and returns
        self._compute_gae()

        time_update_start = self._time(self._advantages.value)

        # Perform PPO update
        update_metrics = self._update()

        time_update_end = self._time(update_metrics)

        # Carry over final observation for next rollout
        self._obs.value = self._obs.value.at[0].set(self._obs.value[self.config.num_steps])

        # Combine all metrics (convert JAX arrays to floats)
        metrics = {k: float(v) for k, v in {**rollout_metrics, **update_metrics}.items()}
        metrics["duration_rollout"] = time_update_start - time_rollout_start
        metrics["duration_update"] = time_update_end - time_update_start

        return metrics

    def train_from_scratch(self) -> None:
        """Run the full PPO training loop."""
        # Use start_time passed from runner for initial overhead, or fallback to now
        time_start = self.start_time if self.start_time is not None else self._time()

        # Duration tracking: first iteration, first 8 (0:7), and rest (7:)
        duration_iteration_0 = 0.0
        duration_iteration_sum_0_7 = 0.0
        duration_iteration_sum_7_plus = 0.0
        duration_rollout_0 = 0.0
        duration_rollout_sum_0_7 = 0.0
        duration_rollout_sum_7_plus = 0.0
        duration_update_0 = 0.0
        duration_update_sum_0_7 = 0.0
        duration_update_sum_7_plus = 0.0

        self.reset()
        self._log_hparams()

        # Progress bar with key metrics
        pbar = tqdm(range(self.config.num_iterations), desc="Training")
        for iteration in pbar:
            time_iteration_start = self._time()

            # TRAINING STEP - the only functional (non-logging) line in the loop
            jax_metrics = self.train_step()

            # Convert JAX arrays to Python floats (GPU -> CPU transfer, triggers sync)
            metrics: dict[str, float] = {k: float(v) for k, v in jax_metrics.items()}

            # Timing
            time_iteration_end = self._time()
            duration_iteration = time_iteration_end - time_iteration_start
            duration_rollout = metrics["duration_rollout"]
            duration_update = metrics["duration_update"]

            # Accumulate durations: first (0), first 8 (0:7), rest (7:)
            if iteration == 0:
                duration_iteration_0 = time_iteration_end - time_start  # From start_time
                duration_rollout_0 = duration_rollout
                duration_update_0 = duration_update
            if iteration < 8:
                duration_iteration_sum_0_7 += duration_iteration
                duration_rollout_sum_0_7 += duration_rollout
                duration_update_sum_0_7 += duration_update
            else:
                duration_iteration_sum_7_plus += duration_iteration
                duration_rollout_sum_7_plus += duration_rollout
                duration_update_sum_7_plus += duration_update

            metrics["duration_iteration"] = duration_iteration
            metrics["time_elapsed"] = time_iteration_end - time_start

            # Log to WandB
            self._log_metrics(metrics, iteration)

            # Update progress bar
            avg_reward = metrics.get("avg_episode_reward", float("nan"))
            approx_kl = metrics.get("approx_kl", float("nan"))

            # Format display values (handle NaN gracefully)
            reward_str = f"{avg_reward:.1f}" if not math.isnan(avg_reward) else "N/A"
            kl_str = f"{approx_kl:.4f}" if not math.isnan(approx_kl) else "N/A"

            pbar.set_postfix({
                "reward": reward_str,
                "kl": kl_str,
            })

        pbar.close()

        # Final timings
        time_end = self._time()
        duration_total = time_end - time_start
        num_0_7 = min(8, self.config.num_iterations)
        num_7_plus = max(1, self.config.num_iterations - 8)
        duration_iteration_avg_0_7 = duration_iteration_sum_0_7 / num_0_7
        duration_iteration_avg_7_plus = duration_iteration_sum_7_plus / num_7_plus if self.config.num_iterations > 8 else 0.0
        duration_rollout_avg_0_7 = duration_rollout_sum_0_7 / num_0_7
        duration_rollout_avg_7_plus = duration_rollout_sum_7_plus / num_7_plus if self.config.num_iterations > 8 else 0.0
        duration_update_avg_0_7 = duration_update_sum_0_7 / num_0_7
        duration_update_avg_7_plus = duration_update_sum_7_plus / num_7_plus if self.config.num_iterations > 8 else 0.0

        self._log_summary({
            "duration_total": duration_total,
            "duration_iteration_0": duration_iteration_0,
            "duration_iteration_avg_0:7": duration_iteration_avg_0_7,
            "duration_iteration_avg_7:": duration_iteration_avg_7_plus,
            "duration_rollout_0": duration_rollout_0,
            "duration_rollout_avg_0:7": duration_rollout_avg_0_7,
            "duration_rollout_avg_7:": duration_rollout_avg_7_plus,
            "duration_update_0": duration_update_0,
            "duration_update_avg_0:7": duration_update_avg_0_7,
            "duration_update_avg_7:": duration_update_avg_7_plus,
        })

        # Final summary
        print(f"\nTraining completed in {duration_total/60:.1f} minutes")
        print(f"Iteration 0: {duration_iteration_0:.3f}s (from start_time)")
        print(f"Iteration avg 0:7: {duration_iteration_avg_0_7:.3f}s, avg 7+: {duration_iteration_avg_7_plus:.3f}s")
        print(f"Rollout 0: {duration_rollout_0:.4f}s, avg 0:7: {duration_rollout_avg_0_7:.4f}s, avg 7+: {duration_rollout_avg_7_plus:.4f}s")
        print(f"Update 0: {duration_update_0:.4f}s, avg 0:7: {duration_update_avg_0_7:.4f}s, avg 7+: {duration_update_avg_7_plus:.4f}s")
        print(f"Final average reward: {reward_str}")
