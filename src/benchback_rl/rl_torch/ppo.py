"""Proximal Policy Optimization (PPO) implementation.

Implements the 13 core implementation details from:
https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/

Design choices:
- done[t] indicates episode ended AFTER taking action[t] (not before observing obs[t])
- All terminations (true terminal states and time-limit truncations) are treated the same,
  which introduces a small bias for truncated episodes but simplifies the implementation
- obs[0:num_steps+1] with the extra slot for the bootstrap observation
"""

import math
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field

import jax
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

from benchback_rl.environment.torch_env import TorchEnv
from benchback_rl.rl_torch.models import ActorCritic


# Type alias for learning rate schedule functions
LRSchedule = Callable[[float], float]


def linear_schedule(progress: float) -> float:
    """Linear learning rate schedule from 1.0 to 0.0.
    
    Args:
        progress: Training progress from 0.0 (start) to 1.0 (end).
    
    Returns:
        Learning rate fraction (1.0 at start, 0.0 at end).
    """
    return 1.0 - progress


@dataclass
class PPOHyperparameters:
    """All hyperparameters for PPO training.
    
    Attributes:
        # Environment dimensions (validated against actual env)
        num_envs: Number of parallel environments.
        obs_dim: Observation dimension.
        
        # Rollout parameters
        num_steps: Number of steps per rollout per environment.
        
        # PPO hyperparameters
        learning_rate: Initial learning rate.
        gamma: Discount factor.
        gae_lambda: GAE lambda parameter.
        clip_coef: PPO clipping coefficient.
        clip_vloss: Whether to clip value function loss.
        ent_coef: Entropy coefficient.
        vf_coef: Value function coefficient.
        max_grad_norm: Maximum gradient norm for clipping.
        
        # Adam parameters
        adam_eps: Adam epsilon.
        adam_betas: Adam beta parameters.
        
        # Training parameters
        num_minibatches: Number of minibatches per update.
        update_epochs: Number of epochs per update.
        num_iterations: Total number of training iterations (rollout + update cycles).
        
        # Learning rate schedule
        lr_schedule: Function mapping progress (0.0 to 1.0) to LR fraction.
        
        # Seeding
        seed: Optional seed for reproducibility (seeds both torch and JAX).
        
        # Benchmarking
        sync_for_timing: Whether to force CUDA synchronization before timing operations.
    """
    # Environment dimensions
    num_envs: int
    obs_dim: int
    
    # Rollout parameters
    num_steps: int = 128
    
    # PPO hyperparameters
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Adam parameters
    adam_eps: float = 1e-5
    adam_betas: tuple[float, float] = (0.9, 0.999)
    
    # Training parameters
    num_minibatches: int = 4
    update_epochs: int = 4
    num_iterations: int = 1000
    
    # Learning rate schedule (default: linear decay to 0)
    lr_schedule: LRSchedule = field(default=linear_schedule)
    
    # Seeding (None = random)
    seed: int | None = None
    
    # Benchmarking
    sync_for_timing: bool = False
    
    @property
    def batch_size(self) -> int:
        """Total batch size per iteration."""
        return self.num_envs * self.num_steps
    
    @property
    def minibatch_size(self) -> int:
        """Size of each minibatch."""
        return self.batch_size // self.num_minibatches
    
    @property
    def total_timesteps(self) -> int:
        """Total environment timesteps over training."""
        return self.num_iterations * self.batch_size


class PPO:
    """Proximal Policy Optimization algorithm.
    
    Storage layout for a rollout of T steps:
        obs[0:T]     - observations fed to the network
        obs[T]       - final observation (for bootstrap value)
        actions[t]   - action taken given obs[t]
        log_probs[t] - log probability of action[t]
        values[t]    - value estimate for obs[t]
        values[T]    - bootstrap value for obs[T]
        rewards[t]   - reward received after taking action[t]
        dones[t]     - whether episode ended after taking action[t]
    """

    def __init__(
        self,
        env: TorchEnv,
        agent: ActorCritic,
        hparams: PPOHyperparameters,
    ) -> None:
        self.device = torch.device("cuda")

        # Validate environment dimensions match config
        if env.num_envs != hparams.num_envs:
            raise ValueError(
                f"Environment num_envs ({env.num_envs}) does not match "
                f"config num_envs ({hparams.num_envs})"
            )
        if env.obs_dim != hparams.obs_dim:
            raise ValueError(
                f"Environment obs_dim ({env.obs_dim}) does not match "
                f"config obs_dim ({hparams.obs_dim})"
            )

        # Store references
        self.hparams = hparams
        self.env = env
        self.agent = agent

        # Detail 3: Adam with eps=1e-5
        self.optimizer = torch.optim.Adam(
            self.agent.parameters(),
            lr=hparams.learning_rate,
            eps=hparams.adam_eps,
            betas=hparams.adam_betas,
        )

        # Rollout storage (pre-allocated on GPU)
        # obs and values have one extra slot for the final observation/bootstrap value
        self._obs = torch.zeros((hparams.num_steps + 1, hparams.num_envs, hparams.obs_dim), device=self.device)
        self._values = torch.zeros((hparams.num_steps + 1, hparams.num_envs), device=self.device)
        self._actions = torch.zeros((hparams.num_steps, hparams.num_envs), dtype=torch.long, device=self.device)
        self._log_probs = torch.zeros((hparams.num_steps, hparams.num_envs), device=self.device)
        self._rewards = torch.zeros((hparams.num_steps, hparams.num_envs), device=self.device)
        self._dones = torch.zeros((hparams.num_steps, hparams.num_envs), dtype=torch.bool, device=self.device)
        self._advantages = torch.zeros((hparams.num_steps, hparams.num_envs), device=self.device)
        self._returns = torch.zeros((hparams.num_steps, hparams.num_envs), device=self.device)

        # Episode statistics tracking (per-env accumulators)
        self._episode_returns = torch.zeros(hparams.num_envs, device=self.device)
        self._episode_lengths = torch.zeros(hparams.num_envs, dtype=torch.long, device=self.device)
        self._last_episode_metrics = PPO.default_episode_metrics()

        # RNG state (will be set by seed())
        self._rng_key: jax.Array

        # Seed everything (use provided seed or generate random one)
        effective_seed = hparams.seed if hparams.seed is not None else int(torch.randint(0, 2**31, (1,)).item())
        self.seed(effective_seed)

    def _time(self) -> float:
        """Get current time, optionally syncing CUDA first for accurate timing."""
        if self.hparams.sync_for_timing:
            torch.cuda.synchronize()
        return time.perf_counter()

    @staticmethod
    def default_episode_metrics() -> dict[str, float]:
        """Return default episode metrics with NaN values."""
        return {
            "episodes_completed": 0,
            "avg_episode_length": float("nan"),
            "avg_episode_return": float("nan"),
            "avg_reward_per_step": float("nan"),
            "min_episode_return": float("nan"),
            "max_episode_return": float("nan"),
            "min_episode_length": float("nan"),
            "max_episode_length": float("nan"),
        }

    def seed(self, seed: int) -> None:
        """Seed all random number generators for reproducibility.
        
        Args:
            seed: Seed value for torch and JAX RNGs.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self._rng_key = jax.random.PRNGKey(seed)

    def _reset(self) -> None:
        """Reset environment and prepare for fresh training using internal RNG."""
        self._rng_key, reset_key = jax.random.split(self._rng_key)
        obs = self.env.reset(reset_key)
        self._obs[0] = obs
        # Reset all episode trackers
        self._episode_returns.zero_()
        self._episode_lengths.zero_()
        self._last_episode_metrics = PPO.default_episode_metrics()

    def _collect_rollout(self) -> dict[str, float]:
        """Collect a rollout of experience.
        
        Returns:
            Dict with episode statistics for episodes that completed during this rollout.
        """
        # Per-rollout accumulators (local, on GPU)
        n_completed = torch.tensor(0, dtype=torch.long, device=self.device)
        sum_returns = torch.tensor(0.0, device=self.device)
        sum_lengths = torch.tensor(0, dtype=torch.long, device=self.device)
        sum_reward_per_step = torch.tensor(0.0, device=self.device)
        min_return = torch.tensor(float("inf"), device=self.device)
        max_return = torch.tensor(float("-inf"), device=self.device)
        min_length = torch.tensor(torch.iinfo(torch.long).max, dtype=torch.long, device=self.device)
        max_length = torch.tensor(0, dtype=torch.long, device=self.device)

        rng_key = self._rng_key
        
        for step in range(self.hparams.num_steps):
            # Split key for this step
            rng_key, step_key = jax.random.split(rng_key)

            # Get current observation
            obs = self._obs[step]

            # Get action and value
            with torch.no_grad():
                action, log_prob, _, value = self.agent.get_action_and_value(obs)

            # Step environment
            next_obs, reward, done, _ = self.env.step(step_key, action)

            # Store transition
            self._actions[step] = action
            self._log_probs[step] = log_prob
            self._values[step] = value
            self._rewards[step] = reward
            self._dones[step] = done
            self._obs[step + 1] = next_obs

            # Track episode statistics (vectorized, all on GPU)
            self._episode_returns += reward
            self._episode_lengths += 1

            if done.any():
                done_returns = self._episode_returns[done]
                done_lengths = self._episode_lengths[done]
                
                n_completed += done.sum()
                sum_returns += done_returns.sum()
                sum_lengths += done_lengths.sum()
                sum_reward_per_step += (done_returns / done_lengths).sum()
                min_return = torch.minimum(min_return, done_returns.min())
                max_return = torch.maximum(max_return, done_returns.max())
                min_length = torch.minimum(min_length, done_lengths.min())
                max_length = torch.maximum(max_length, done_lengths.max())
                
                # Reset completed episodes
                self._episode_returns[done] = 0
                self._episode_lengths[done] = 0

        # Store updated RNG key
        self._rng_key = rng_key

        # Compute bootstrap value
        with torch.no_grad():
            final_value = self.agent.get_value(self._obs[self.hparams.num_steps])
        self._values[self.hparams.num_steps] = final_value

        # Compute GAE
        self._compute_gae()

        # Update last known metrics if episodes completed (CPU transfer happens here)
        episodes_completed = n_completed.item()
        if episodes_completed > 0:
            self._last_episode_metrics = {
                "episodes_completed": episodes_completed,
                "avg_episode_length": (sum_lengths.float() / n_completed).item(),
                "avg_episode_return": (sum_returns / n_completed).item(),
                "avg_reward_per_step": (sum_reward_per_step / n_completed).item(),
                "min_episode_return": min_return.item(),
                "max_episode_return": max_return.item(),
                "min_episode_length": float(min_length.item()),
                "max_episode_length": float(max_length.item()),
            }

        return self._last_episode_metrics

    def _compute_gae(self) -> None:
        """Compute Generalized Advantage Estimation (GAE).
        
        Populates self._advantages and self._returns in-place.
        """
        last_gae_lam = torch.zeros(self.hparams.num_envs, device=self.device)

        for t in reversed(range(self.hparams.num_steps)):
            # done[t] indicates episode ended after action[t], so next state is from new episode
            next_non_terminal = (~self._dones[t]).float()
            delta = (
                self._rewards[t]
                + self.hparams.gamma * self._values[t + 1] * next_non_terminal
                - self._values[t]
            )
            last_gae_lam = delta + self.hparams.gamma * self.hparams.gae_lambda * next_non_terminal * last_gae_lam
            self._advantages[t] = last_gae_lam

        # TD(lambda) returns: advantages + values
        self._returns[:] = self._advantages + self._values[:self.hparams.num_steps]

    def _get_batches(self) -> Iterator[tuple[torch.Tensor, ...]]:
        """Yield shuffled minibatches of flattened rollout data."""
        total_size = self.hparams.batch_size

        # Flatten all tensors (views if contiguous)
        flat_obs = self._obs[:self.hparams.num_steps].reshape(total_size, self.hparams.obs_dim)
        flat_actions = self._actions.reshape(total_size)
        flat_log_probs = self._log_probs.reshape(total_size)
        flat_advantages = self._advantages.reshape(total_size)
        flat_returns = self._returns.reshape(total_size)
        flat_values = self._values[:self.hparams.num_steps].reshape(total_size)

        # Generate shuffled indices
        indices = torch.randperm(total_size, device=self.device)

        # Yield minibatches
        for start in range(0, total_size, self.hparams.minibatch_size):
            batch_indices = indices[start:start + self.hparams.minibatch_size]
            yield (
                flat_obs[batch_indices],
                flat_actions[batch_indices],
                flat_log_probs[batch_indices],
                flat_advantages[batch_indices],
                flat_returns[batch_indices],
                flat_values[batch_indices],
            )

    def _update(self, lr_fraction: float = 1.0) -> dict[str, float]:
        """Perform PPO update on collected rollout.
        
        Args:
            lr_fraction: Fraction for learning rate annealing (1.0 = full LR, 0.0 = no LR)
        
        Returns:
            Dict with update metrics
        """
        # Detail 4: Learning rate annealing
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.hparams.learning_rate * lr_fraction

        # Accumulate metrics across all updates
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_loss = 0.0
        total_approx_kl = 0.0
        total_clip_frac = 0.0
        num_updates = 0

        # Detail 6: Mini-batch updates over multiple epochs
        for _ in range(self.hparams.update_epochs):
            for batch in self._get_batches():
                b_obs, b_actions, b_log_probs, b_advantages, b_returns, b_values = batch

                # Get current policy outputs
                _, new_log_prob, entropy, new_value = self.agent.get_action_and_value(
                    b_obs, b_actions
                )

                log_ratio = new_log_prob - b_log_probs
                ratio = log_ratio.exp()

                # Detail 12: Debug variables - approx KL
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clip_frac = ((ratio - 1.0).abs() > self.hparams.clip_coef).float().mean()

                # Detail 7: Advantage normalization (per minibatch)
                mb_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

                # Detail 8: Clipped surrogate objective
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.hparams.clip_coef, 1 + self.hparams.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Detail 9: Value function loss clipping
                if self.hparams.clip_vloss:
                    v_loss_unclipped = (new_value - b_returns) ** 2
                    v_clipped = b_values + torch.clamp(
                        new_value - b_values, -self.hparams.clip_coef, self.hparams.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((new_value - b_returns) ** 2).mean()

                # Detail 10: Overall loss with entropy bonus
                entropy_loss = entropy.mean()
                loss = pg_loss - self.hparams.ent_coef * entropy_loss + self.hparams.vf_coef * v_loss

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Detail 11: Global gradient clipping
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.hparams.max_grad_norm)

                self.optimizer.step()

                # Accumulate metrics
                total_policy_loss += pg_loss.item()
                total_value_loss += v_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_loss += loss.item()
                total_approx_kl += approx_kl.item()
                total_clip_frac += clip_frac.item()
                num_updates += 1

        # Compute explained variance
        with torch.no_grad():
            flat_values = self._values[:self.hparams.num_steps].reshape(-1)
            flat_returns = self._returns.reshape(-1)
            explained_var = (1 - (flat_returns - flat_values).var() / flat_returns.var()).item()

        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy_loss": total_entropy_loss / num_updates,
            "total_loss": total_loss / num_updates,
            "approx_kl": total_approx_kl / num_updates,
            "clip_frac": total_clip_frac / num_updates,
            "explained_variance": explained_var,
        }

    def _train_step(self, lr_fraction: float = 1.0) -> dict[str, float]:
        """Perform one complete training step (rollout + update).
        
        This is the function that will be timed for benchmarking.
        Equivalent to the jitted function in JAX implementations.
        
        Args:
            lr_fraction: Learning rate fraction for annealing
        
        Returns:
            Dict with combined metrics from rollout and update, plus timing info
        """
        t0 = self._time()
        
        # Collect rollout (also computes episode stats)
        episode_metrics = self._collect_rollout()
        
        t1 = self._time()
        
        # Perform update
        update_metrics = self._update(lr_fraction)
        
        t2 = self._time()

        # Carry over final observation for next rollout
        self._obs[0] = self._obs[self.hparams.num_steps]

        metrics = episode_metrics | update_metrics
        metrics["time_rollout"] = t1 - t0
        metrics["time_update"] = t2 - t1
        return metrics

    def _log_metrics(self, metrics: dict[str, float], iteration: int) -> None:
        """Log metrics to WandB if a run is active."""
        if wandb.run is None:
            return
        wandb.log({"iteration": iteration, **metrics}, step=iteration)

    def _log_hparams(self) -> None:
        """Log hyperparameters to WandB if a run is active."""
        if wandb.run is None:
            return
        hparams_dict = asdict(self.hparams)
        hparams_dict.pop("lr_schedule", None)  # Not serializable
        wandb.config.update(hparams_dict)

    def train(self) -> None:
        """Run the full PPO training loop."""
        # Reset environment
        self._reset()
        
        # Log hyperparameters to WandB
        self._log_hparams()
        
        # Training metrics
        start_time = self._time()
        
        # Start timing for first iteration's overhead (captures init/reset time)
        last_step_end = self._time()
        
        # Progress bar with key metrics
        pbar = tqdm(range(self.hparams.num_iterations), desc="Training")
        
        for iteration in pbar:
            # Compute learning rate fraction based on progress
            progress = iteration / self.hparams.num_iterations
            lr_fraction = self.hparams.lr_schedule(progress)
            
            # Measure overhead (time between end of last step and start of this one)
            step_start = self._time()
            overhead = step_start - last_step_end
            
            # Perform training step
            metrics = self._train_step(lr_fraction)
            
            # Record end of step for next iteration's overhead calculation
            last_step_end = self._time()
            
            # Add overhead and total step time to metrics
            metrics["time_overhead"] = overhead
            metrics["time_step"] = last_step_end - step_start
            
            # Compute steps per second and elapsed time
            elapsed = last_step_end - start_time
            total_steps = (iteration + 1) * self.hparams.batch_size
            sps = total_steps / elapsed if elapsed > 0 else 0
            
            # Add timing metrics
            metrics["sps"] = sps
            metrics["learning_rate"] = self.hparams.learning_rate * lr_fraction
            metrics["time_elapsed"] = elapsed
            
            # Log to WandB
            self._log_metrics(metrics, iteration)
            
            # Update progress bar
            avg_return = metrics.get("avg_episode_return", float("nan"))
            approx_kl = metrics.get("approx_kl", float("nan"))
            
            # Format display values (handle NaN gracefully)
            return_str = f"{avg_return:.1f}" if not math.isnan(avg_return) else "N/A"
            kl_str = f"{approx_kl:.4f}" if not math.isnan(approx_kl) else "N/A"
            
            pbar.set_postfix({
                "SPS": f"{sps:.0f}",
                "return": return_str,
                "kl": kl_str,
            })
        
        pbar.close()
        
        # Final summary
        total_time = self._time() - start_time
        total_steps = self.hparams.num_iterations * self.hparams.batch_size
        print(f"\nTraining completed in {total_time:.1f}s")
        print(f"Total timesteps: {total_steps:,}")
        print(f"Average SPS: {total_steps / total_time:.0f}")
