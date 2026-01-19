from typing import Any
import math
import time

import jax
import jax.numpy as jnp
import optax
import wandb
from tqdm import tqdm

from benchback_rl.rl_common import PPOConfig
from benchback_rl.rl_linen.models import ActorCritic, ModelParams
from benchback_rl.rl_linen.ppo.rollout import EnvCarry, RolloutWithGAE, collect_rollout, compute_gae
from benchback_rl.rl_linen.ppo.update import update, TrainState

# Type aliases for gymnax environments
Env = Any
EnvParams = Any


class PPO:
    """Proximal Policy Optimization algorithm (JAX/Flax Linen).
    
    Stateful class that wraps jitted rollout and update functions.
    API mirrors the torch implementation for easy comparison.
    """

    def __init__(
        self,
        env: Env,
        env_params: EnvParams,
        model: ActorCritic,
        model_params: ModelParams,
        config: PPOConfig,
        start_time: float | None = None,
    ) -> None:
        """Initialize PPO trainer.
        
        Note: Call reset() before train_step() or use train_from_scratch().
        
        Args:
            env: Gymnax environment (single, not vmapped).
            env_params: Gymnax environment parameters.
            model: ActorCritic Flax module.
            model_params: Initial model parameters.
            config: PPO configuration.
            start_time: Time at beginning of main for initial overhead calculation.
        """
        self.config = config
        self.env = env
        self.env_params = env_params
        self.model = model
        self.start_time = start_time

        # Verify config matches expected framework
        if config.framework != "linen":
            raise ValueError(f"Expected framework='linen', got '{config.framework}'")

        # Create vmapped environment functions (not JIT compiled here - compiled from above in the rollout)
        self._env_reset = jax.vmap(env.reset, in_axes=(0, None))
        self._env_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

        # Create optimizer with linear LR schedule annealing to 0
        schedule = optax.linear_schedule(
            init_value=config.learning_rate,
            end_value=0.0,
            transition_steps=config.total_optimizer_steps,
        )
        self._optimizer = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(
                learning_rate=schedule,
                eps=config.adam_eps,
                b1=config.adam_betas[0],
                b2=config.adam_betas[1],
            ),  
        )   
    
        # Create initial train state    
        self._train_state = TrainState( 
            model_params=model_params,  
            opt_state=self._optimizer.init(model_params),
            step=jnp.array(0, dtype=jnp.int32),
        )

        # Initialize RNG (use provided seed or generate one based on time)
        seed = config.seed if config.seed is not None else int(time.time_ns())
        self._rng_key = jax.random.PRNGKey(seed)

        # Environment carry state (initialized by reset())
        self._env_carry: EnvCarry

        # JIT compile core functions based on config.compile
        if config.compile == "jax.jit":
            self._collect_rollout_fn = jax.jit(
                collect_rollout,
                static_argnames=("config", "env_step_fn", "model"),
            )
            self._compute_gae_fn = jax.jit(
                compute_gae,
                static_argnames=("gamma", "gae_lambda"),
            )
            self._update_fn = jax.jit(
                update,
                static_argnames=("config", "model", "optimizer"),
            )
        else:
            # No JIT (compile == "none")
            self._collect_rollout_fn = collect_rollout
            self._compute_gae_fn = compute_gae
            self._update_fn = update

    def _time(self, block_until_ready_object: Any|None = None) -> float:
        """Get current time, optionally syncing JAX first for accurate timing."""
        if self.config.sync_for_timing and block_until_ready_object is not None:
            jax.block_until_ready(block_until_ready_object)
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

    def reset(self) -> None:
        """Reset environment and prepare for fresh training.
        
        Must be called before train_step(). Called automatically by train_from_scratch().
        """
        # Split key for environment reset
        self._rng_key, reset_rng_key = jax.random.split(self._rng_key)
        reset_rng_keys = jax.random.split(reset_rng_key, self.config.num_envs)

        # Reset vectorized environments
        obs, env_state = self._env_reset(reset_rng_keys, self.env_params)

        # Initialize environment carry state
        self._env_carry = EnvCarry.from_reset(obs, env_state)

    def train_step(self) -> dict[str, float]:
        """Perform one complete training step (rollout + update).
        
        This is the function that will be timed for benchmarking.
        Equivalent to the jitted function in JAX implementations.
        
        Returns:
            Dict with combined metrics from rollout and update, plus timing info.
        """
        # Sync on env_carry.obs to ensure previous iteration's updates are complete
        time_rollout_start = self._time(self._env_carry.obs)

        # Split RNG key for rollout
        self._rng_key, rollout_rng_key = jax.random.split(self._rng_key)

        # Collect rollout (without GAE)
        (obs, actions, log_probs, values, rewards, dones,
         final_obs, final_value, new_env_carry, episode_metrics) = self._collect_rollout_fn(
            config=self.config,
            env_step_fn=self._env_step,
            env_params=self.env_params,
            env_carry=self._env_carry,
            model=self.model,
            model_params=self._train_state.model_params,
            rng_key=rollout_rng_key,
        )

        # Compute GAE
        advantages, returns = self._compute_gae_fn(
            values=values,
            rewards=rewards,
            dones=dones,
            final_value=final_value,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

        # Build RolloutWithGAE dataclass for update
        rollout_data = RolloutWithGAE(
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

        time_update_start = self._time(rollout_data)

        # Split RNG key for update
        self._rng_key, update_rng_key = jax.random.split(self._rng_key)

        # Perform PPO update
        new_train_state, update_metrics = self._update_fn(
            config=self.config,
            model=self.model,
            optimizer=self._optimizer,
            train_state=self._train_state,
            rollout_with_gae=rollout_data,
            rng_key=update_rng_key,
        )

        time_update_end = self._time(new_train_state)

        # Update mutable state
        self._env_carry = new_env_carry
        self._train_state = new_train_state

        # Combine metrics and convert to Python floats (GPU -> CPU transfer)
        metrics: dict[str, float] = {}
        
        # Episode metrics from rollout
        for key, value in episode_metrics.items():
            metrics[key] = float(value)
        
        # Update metrics
        for key, value in update_metrics.items():
            metrics[key] = float(value)

        # Timing metrics
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
            metrics = self.train_step()

            # Timing
            time_iteration_end = self._time(metrics)
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
