"""Proximal Policy Optimization (PPO) implementation.

Implements the 13 core implementation details from:
https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/

Design choices:
- done[t] indicates episode ended AFTER taking action[t] (not before observing obs[t])
- All terminations (true terminal states and time-limit truncations) are treated the same,
  which introduces a small bias for truncated episodes but simplifies the implementation
- obs[0:num_steps+1] with the extra slot for the bootstrap observation
"""
from typing import Any
from collections.abc import Iterator

import math
import time
import dataclasses
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import wandb
from tqdm import tqdm

from benchback_rl.rl_torch.env import TorchEnv
from benchback_rl.rl_torch.models import ActorCritic
from benchback_rl.rl_common.config import PPOConfig
from dataclasses import dataclass, field

@dataclass
class EpisodeMetrics:
    """Episode statistics collected during rollouts. Dataclass needed for defaults."""
    episodes_completed: int = 0
    avg_episode_length: float = field(default_factory=lambda: float("nan"))
    avg_episode_reward: float = field(default_factory=lambda: float("nan"))
    avg_reward_per_step: float = field(default_factory=lambda: float("nan"))
    min_episode_reward: float = field(default_factory=lambda: float("nan"))
    max_episode_reward: float = field(default_factory=lambda: float("nan"))
    min_episode_length: float = field(default_factory=lambda: float("nan"))
    max_episode_length: float = field(default_factory=lambda: float("nan"))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


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
        config: PPOConfig,
    ) -> None:
        
        # Verify config matches expected framework
        if config.framework != "torch":
            raise ValueError(f"Expected framework='torch', got '{config.framework}'")

        self.device = torch.device("cuda")

        # Store arguments
        self.config = config
        self.env = env
        self.agent = agent

        # Verify and store environment dimensions
        if env.num_envs != config.num_envs:
            raise ValueError(
                f"Environment num_envs ({env.num_envs}) does not match "
                f"config num_envs ({config.num_envs})"
            )
        self.obs_dim = env.obs_dim

        # Optimiser: Adam
        self.optimizer = torch.optim.Adam(
            self.agent.parameters(),
            lr=config.learning_rate,
            eps=config.adam_eps,
            betas=config.adam_betas,
        )

        # Learning rate annealing (linear decay to 0 over all optimizer steps)
        total_steps = config.total_optimizer_steps
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: 1.0 - step / total_steps,
        )

        # Rollout storage (pre-allocated on GPU)
        # obs and values have one extra slot for the final observation/bootstrap value
        self._obs = torch.zeros((config.num_steps + 1, config.num_envs, self.obs_dim), device=self.device)
        self._values = torch.zeros((config.num_steps + 1, config.num_envs), device=self.device)
        self._actions = torch.zeros((config.num_steps, config.num_envs), dtype=torch.long, device=self.device)
        self._log_probs = torch.zeros((config.num_steps, config.num_envs), device=self.device)
        self._rewards = torch.zeros((config.num_steps, config.num_envs), device=self.device)
        self._dones = torch.zeros((config.num_steps, config.num_envs), dtype=torch.bool, device=self.device)
        self._advantages = torch.zeros((config.num_steps, config.num_envs), device=self.device)
        self._returns = torch.zeros((config.num_steps, config.num_envs), device=self.device)

        # Episode statistics tracking (per-env accumulators)
        self._episode_rewards = torch.zeros(config.num_envs, device=self.device)
        self._episode_lengths = torch.zeros(config.num_envs, dtype=torch.long, device=self.device)

        # Random seed everything (use provided seed or generate one based on time)
        seed = self.config.seed if self.config.seed is not None else int(time.time_ns())
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _time(self) -> float:
        """Get current time, optionally syncing CUDA first for accurate timing."""
        if self.config.sync_for_timing:
            torch.cuda.synchronize()
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

    @torch.no_grad
    def _collect_rollout(self) -> EpisodeMetrics:
        """Collect a rollout of experience.
        
        Returns:
            EpisodeMetrics with episode statistics for episodes that completed during this rollout.
        """
        # Per-rollout accumulators (local, on GPU)
        n_completed = torch.tensor(0, dtype=torch.long, device=self.device)
        sum_rewards = torch.tensor(0.0, device=self.device)
        sum_lengths = torch.tensor(0, dtype=torch.long, device=self.device)
        sum_reward_per_step = torch.tensor(0.0, device=self.device)
        min_rewards = torch.tensor(float("inf"), device=self.device)
        max_rewards = torch.tensor(float("-inf"), device=self.device)
        min_length = torch.tensor(float("inf"), device=self.device)
        max_length = torch.tensor(float("-inf"), device=self.device)
        
        for step in range(self.config.num_steps):
            # Get current observation
            obs = self._obs[step]

            # Get action and value (torch.no_grad is importatnt)
            action, log_prob, _, value = self.agent.get_action_and_value(obs)

            # Step environment
            next_obs, reward, done, _ = self.env.step(action)

            # Store transition
            self._actions[step] = action
            self._log_probs[step] = log_prob
            self._values[step] = value
            self._rewards[step] = reward
            self._dones[step] = done
            self._obs[step + 1] = next_obs

            # Track episode statistics (vectorized, all on GPU)
            self._episode_rewards += reward
            self._episode_lengths += 1

            # Update metrics accumulators for completed episodes (no .any() to avoid sync)
            # Following the JAX pattern: always compute, mask handles the "no done" case
            n_completed += done.sum().to(torch.long)
            sum_rewards += (self._episode_rewards * done).sum()
            sum_lengths += (self._episode_lengths * done).sum()
            sum_reward_per_step += (self._episode_rewards / self._episode_lengths * done).sum()
            min_rewards = torch.minimum(min_rewards,
                                        torch.where(done, self._episode_rewards, torch.full_like(self._episode_rewards, float("inf"))).min())
            max_rewards = torch.maximum(max_rewards,
                                        torch.where(done, self._episode_rewards, torch.full_like(self._episode_rewards, float("-inf"))).max())
            min_length = torch.minimum(min_length,
                                       torch.where(done, self._episode_lengths.float(), torch.full((self.config.num_envs,), float("inf"), device=self.device)).min())
            max_length = torch.maximum(max_length,
                                       torch.where(done, self._episode_lengths.float(), torch.full((self.config.num_envs,), float("-inf"), device=self.device)).max())
            
            # Reset completed episodes (masked assignment)
            self._episode_rewards = torch.where(done, torch.zeros_like(self._episode_rewards), self._episode_rewards)
            self._episode_lengths = torch.where(done, torch.zeros_like(self._episode_lengths), self._episode_lengths)


        # Compute bootstrap value
        final_value = self.agent.get_value(self._obs[self.config.num_steps])
        self._values[self.config.num_steps] = final_value

        # Compute GAE
        self._compute_gae()

        # Build episode metrics (NaN values if no episodes completed)
        episodes_completed = n_completed.item()
        if episodes_completed > 0:
            return EpisodeMetrics(
                episodes_completed=int(episodes_completed),
                avg_episode_length=(sum_lengths.float() / n_completed).item(),
                avg_episode_reward=(sum_rewards / n_completed).item(),
                avg_reward_per_step=(sum_reward_per_step / n_completed).item(),
                min_episode_reward=min_rewards.item(),
                max_episode_reward=max_rewards.item(),
                min_episode_length=min_length.item(),
                max_episode_length=max_length.item(),
            )
        else:
            # Return default metrics with NaN values
            return EpisodeMetrics()
    
    @torch.no_grad
    def _compute_gae(self) -> None:
        """Compute Generalized Advantage Estimation (GAE).
        
        Populates self._advantages and self._returns in-place.
        """
        next_gae = torch.zeros(self.config.num_envs, device=self.device)

        for t in reversed(range(self.config.num_steps)):
            # done[t] indicates episode ended after action[t], so next state is from new episode
            not_done = (~self._dones[t]).float()
            delta = (
                self._rewards[t]
                + self.config.gamma * self._values[t + 1] * not_done
                - self._values[t]
            )
            next_gae = delta + self.config.gamma * self.config.gae_lambda * not_done * next_gae
            self._advantages[t] = next_gae

        # TD(lambda) returns: advantages + values
        self._returns[:] = self._advantages + self._values[:self.config.num_steps]

    @torch.no_grad
    def _get_batches(self) -> Iterator[tuple[torch.Tensor, ...]]:
        """Yield shuffled minibatches of flattened rollout data."""
        total_size = self.config.batch_size

        # Flatten all tensors (views if contiguous)
        flat_obs = self._obs[:self.config.num_steps].reshape(total_size, self.obs_dim)
        flat_actions = self._actions.reshape(total_size)
        flat_log_probs = self._log_probs.reshape(total_size)
        flat_advantages = self._advantages.reshape(total_size)
        flat_returns = self._returns.reshape(total_size)
        flat_values = self._values[:self.config.num_steps].reshape(total_size)

        # Generate shuffled indices
        indices = torch.randperm(total_size, device=self.device)

        # Yield minibatches
        for start in range(0, total_size, self.config.minibatch_size):
            batch_indices = indices[start:start + self.config.minibatch_size]
            yield (
                flat_obs[batch_indices],
                flat_actions[batch_indices],
                flat_log_probs[batch_indices],
                flat_advantages[batch_indices],
                flat_returns[batch_indices],
                flat_values[batch_indices],
            )

    def _update(self) -> dict[str, float]:
        """Perform PPO update on collected rollout.
        
        Returns:
            Dict with update metrics
        """
        # Accumulate metrics across all updates
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_loss = 0.0
        total_approx_kl = 0.0
        total_clip_frac = 0.0
        num_updates = 0

        # Detail 6: Mini-batch updates over multiple epochs
        for _ in range(self.config.update_epochs):
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
                    clip_frac = ((ratio - 1.0).abs() > self.config.clip_coef).float().mean()

                # Detail 7: Advantage normalization (per minibatch)
                mb_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

                # Detail 8: Clipped surrogate objective
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Detail 9: Value function loss clipping
                if self.config.clip_vloss:
                    v_loss_unclipped = (new_value - b_returns) ** 2
                    v_clipped = b_values + torch.clamp(
                        new_value - b_values, -self.config.clip_coef, self.config.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((new_value - b_returns) ** 2).mean()

                # Detail 10: Overall loss with entropy bonus
                entropy_loss = entropy.mean()
                loss = pg_loss - self.config.ent_coef * entropy_loss + self.config.vf_coef * v_loss

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Detail 11: Global gradient clipping
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.max_grad_norm)

                self.optimizer.step()
                self.scheduler.step()

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
            flat_values = self._values[:self.config.num_steps].reshape(-1)
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

    def train_step(self) -> dict[str, float]:
        """Perform one complete training step (rollout + update).
        
        This is the function that will be timed for benchmarking.
        Equivalent to the jitted function in JAX implementations.
        
        Returns:
            Dict with combined metrics from rollout and update, plus timing info
        """
        time_rollout_start = self._time()
        
        # Collect rollout (also computes episode stats)
        episode_metrics = self._collect_rollout()
        
        time_rollout_end = self._time()
        
        # Perform update
        update_metrics = self._update()
        
        time_update_end = self._time()

        # Carry over final observation for next rollout
        self._obs[0] = self._obs[self.config.num_steps]

        metrics = episode_metrics.to_dict() | update_metrics
        metrics["duration_rollout"] = time_rollout_end - time_rollout_start
        metrics["duration_update"] = time_update_end - time_rollout_end
        return metrics

    def reset(self) -> None:
        """Reset environment and prepare for fresh training using internal RNG."""
        obs = self.env.reset()  # reuses the original seed
        self._obs[0] = obs
        # Reset episode trackers
        self._episode_rewards.zero_()
        self._episode_lengths.zero_()

    def train_from_scratch(self) -> None:
        """Run the full PPO training loop."""
        time_start = self._time()
        time_step_end = time_start # for the initial overhead timing

        duration_first_step = 0
        duration_first_overhead = 0
        duration_second_plus_step_sum = 0
        duration_second_plus_overhead_sum = 0
        
        self.reset()
        self._log_hparams()
        
        # Progress bar with key metrics
        pbar = tqdm(range(self.config.num_iterations), desc="Training")
        for iteration in pbar:
            # Timing
            time_iteration_start = self._time()  # initialise for first reading to log overhead
            duration_overhead = time_iteration_start - time_step_end
            if iteration == 0:
                duration_first_overhead = duration_overhead
            else:
                duration_second_plus_overhead_sum += duration_overhead

            # TRAINING STEP - the only functional (non-logging) line in the loop
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
        
        # Log to WandB - final timings
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

        # Final summary
        print(f"\nTraining completed in {duration_total/60:.1f} minutes")
        print(f"First iteration time: {duration_first_step+duration_first_overhead:.3f}s ")
        print(f"Average time per iteration (excluding first): "
              f"{duration_average_second_plus_step+duration_average_second_plus_overhead:.3f}s ")
        print(f"Final average reward: {reward_str}")
