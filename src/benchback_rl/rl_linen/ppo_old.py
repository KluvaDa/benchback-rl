"""Proximal Policy Optimization (PPO) implementation using JAX and Flax Linen.

Implements the 13 core implementation details from:
https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import asdict
from typing import Any

import flax.struct
import gymnax
import jax
import jax.numpy as jnp
import optax
import wandb
from tqdm import tqdm

from benchback_rl.rl_common.config import PPOConfig
from benchback_rl.rl_linen.models import ActorCritic, DefaultActorCritic

# Type aliases
Params = dict[str, Any]
EnvState = Any
EnvParams = Any

# === State Containers ===

@flax.struct.dataclass
class PPOState:
    """Complete training state passed through train_step."""

    params: Params
    opt_state: optax.OptState
    env_state: EnvState
    obs: jax.Array  # (num_envs, obs_dim)
    episode_returns: jax.Array  # (num_envs,) running totals
    episode_lengths: jax.Array  # (num_envs,) running totals
    rng_key: jax.Array


@flax.struct.dataclass
class Rollout:
    """Raw rollout data before flattening."""

    obs: jax.Array  # (num_steps, num_envs, obs_dim)
    actions: jax.Array  # (num_steps, num_envs)
    log_probs: jax.Array  # (num_steps, num_envs)
    values: jax.Array  # (num_steps, num_envs)
    rewards: jax.Array  # (num_steps, num_envs)
    dones: jax.Array  # (num_steps, num_envs)


@flax.struct.dataclass
class EpisodeMetrics:
    """Episode statistics accumulated during rollout."""

    completed: jax.Array  # scalar count
    sum_returns: jax.Array  # scalar
    sum_lengths: jax.Array  # scalar
    sum_reward_per_step: jax.Array  # scalar
    min_return: jax.Array  # scalar
    max_return: jax.Array  # scalar
    min_length: jax.Array  # scalar
    max_length: jax.Array  # scalar


@flax.struct.dataclass
class StepMetrics:
    """Output metrics from train_step for logging."""

    # Episode metrics
    episodes_completed: jax.Array
    avg_return: jax.Array
    avg_length: jax.Array
    avg_reward_per_step: jax.Array
    min_return: jax.Array
    max_return: jax.Array
    min_length: jax.Array
    max_length: jax.Array
    # Loss metrics
    policy_loss: jax.Array
    value_loss: jax.Array
    entropy: jax.Array
    total_loss: jax.Array
    approx_kl: jax.Array
    clip_frac: jax.Array
    explained_variance: jax.Array


# === PPO Algorithm ===


class PPO:
    """Proximal Policy Optimization with JIT-compiled train_step."""

    def __init__(self, config: PPOConfig) -> None:
        if config.framework != "linen":
            raise ValueError(f"Expected framework='linen', got '{config.framework}'")

        self.config = config
        self.hparams = config.hparams

        self._init_env()
        self._init_model()
        self._init_optimizer()
        self._train_step_jit = jax.jit(self._train_step)

    def _init_env(self) -> None:
        """Initialize gymnax environment and vectorized functions."""
        env, env_params = gymnax.make(self.config.env_name)
        self.env = env
        self.env_params = env_params

        obs_space = env.observation_space(env_params)  # type: ignore[arg-type]
        action_space = env.action_space(env_params)  # type: ignore[arg-type]
        self.obs_dim = int(obs_space.shape[0])  # type: ignore[union-attr]
        self.action_dim = int(action_space.n)  # type: ignore[union-attr]

        self._reset_fn = jax.vmap(env.reset, in_axes=(0, None))
        self._step_fn = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    def _init_model(self) -> None:
        """Initialize ActorCritic model."""
        self.model = DefaultActorCritic(action_dim=self.action_dim)

    def _init_optimizer(self) -> None:
        """Initialize optimizer with linear LR schedule."""
        schedule = optax.linear_schedule(
            init_value=self.hparams.learning_rate,
            end_value=0.0,
            transition_steps=self.config.total_optimizer_steps,
        )
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(self.hparams.max_grad_norm),
            optax.adam(
                learning_rate=schedule,
                eps=self.hparams.adam_eps,
                b1=self.hparams.adam_betas[0],
                b2=self.hparams.adam_betas[1],
            ),
        )

    # === Static methods for pure computations ===

    @staticmethod
    def _gae_step(
        carry: tuple[jax.Array, jax.Array],
        transition: tuple[jax.Array, jax.Array, jax.Array],
        gamma: float,
        gae_lambda: float,
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        """Single step of GAE computation (reverse scan)."""
        last_gae_lam, next_value = carry
        value, reward, done = transition

        next_non_terminal = 1.0 - done.astype(jnp.float32)
        delta = reward + gamma * next_value * next_non_terminal - value
        last_gae_lam = (
            delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        )
        return (last_gae_lam, value), last_gae_lam

    @staticmethod
    def _compute_gae(
        values: jax.Array,
        rewards: jax.Array,
        dones: jax.Array,
        bootstrap_value: jax.Array,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[jax.Array, jax.Array]:
        """Compute GAE advantages and returns via reverse scan."""
        num_envs = bootstrap_value.shape[0]
        init_carry = (jnp.zeros(num_envs), bootstrap_value)

        step_fn = lambda c, t: PPO._gae_step(c, t, gamma, gae_lambda)
        _, advantages = jax.lax.scan(
            step_fn,
            init_carry,
            (values, rewards, dones),
            reverse=True,
        )
        returns = advantages + values
        return advantages, returns

    @staticmethod
    def _ppo_loss(
        params: Params,
        model: ActorCritic,
        obs: jax.Array,
        actions: jax.Array,
        old_log_probs: jax.Array,
        advantages: jax.Array,
        returns: jax.Array,
        old_values: jax.Array,
        clip_coef: float,
        vf_coef: float,
        ent_coef: float,
        clip_vloss: bool,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Compute PPO loss with clipped surrogate objective."""
        output = model.apply(params, obs)
        logits: jax.Array = output[0]  # type: ignore[index]
        new_value: jax.Array = output[1]  # type: ignore[index]

        # Policy loss
        log_probs_all = jax.nn.log_softmax(logits)
        probs = jax.nn.softmax(logits)
        new_log_prob = jnp.take_along_axis(
            log_probs_all, actions[:, None], axis=-1
        ).squeeze(-1)
        entropy = -jnp.sum(probs * log_probs_all, axis=-1)

        log_ratio = new_log_prob - old_log_probs
        ratio = jnp.exp(log_ratio)

        # Debug metrics
        approx_kl = ((ratio - 1) - log_ratio).mean()
        clip_frac = ((jnp.abs(ratio - 1.0) > clip_coef).astype(jnp.float32)).mean()

        # Advantage normalization (per minibatch)
        norm_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Clipped surrogate objective
        pg_loss1 = -norm_advantages * ratio
        pg_loss2 = -norm_advantages * jnp.clip(ratio, 1 - clip_coef, 1 + clip_coef)
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

        # Value loss
        if clip_vloss:
            v_loss_unclipped = (new_value - returns) ** 2
            v_clipped = old_values + jnp.clip(
                new_value - old_values, -clip_coef, clip_coef
            )
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss = 0.5 * jnp.maximum(v_loss_unclipped, v_loss_clipped).mean()
        else:
            v_loss = 0.5 * ((new_value - returns) ** 2).mean()

        entropy_mean = entropy.mean()
        total_loss = pg_loss - ent_coef * entropy_mean + vf_coef * v_loss

        metrics = {
            "policy_loss": pg_loss,
            "value_loss": v_loss,
            "entropy": entropy_mean,
            "total_loss": total_loss,
            "approx_kl": approx_kl,
            "clip_frac": clip_frac,
        }
        return total_loss, metrics

    @staticmethod
    def _update_episode_metrics(
        metrics: EpisodeMetrics,
        done: jax.Array,
        episode_returns: jax.Array,
        episode_lengths: jax.Array,
    ) -> EpisodeMetrics:
        """Update episode metrics for completed episodes."""
        n_done = done.sum()
        done_returns = jnp.where(done, episode_returns, 0.0)
        done_lengths = jnp.where(done, episode_lengths, 0)

        return EpisodeMetrics(
            completed=metrics.completed + n_done,
            sum_returns=metrics.sum_returns + done_returns.sum(),
            sum_lengths=metrics.sum_lengths + done_lengths.sum(),
            sum_reward_per_step=metrics.sum_reward_per_step
            + jnp.where(
                done,
                episode_returns / episode_lengths.astype(jnp.float32),
                0.0,
            ).sum(),
            min_return=jnp.minimum(
                metrics.min_return,
                jnp.where(done.any(), jnp.where(done, done_returns, jnp.inf).min(), jnp.inf),
            ),
            max_return=jnp.maximum(
                metrics.max_return,
                jnp.where(done.any(), jnp.where(done, done_returns, -jnp.inf).max(), -jnp.inf),
            ),
            min_length=jnp.minimum(
                metrics.min_length,
                jnp.where(
                    done.any(),
                    jnp.where(done, done_lengths, jnp.iinfo(jnp.int32).max).min(),
                    jnp.iinfo(jnp.int32).max,
                ),
            ).astype(jnp.float32),
            max_length=jnp.maximum(
                metrics.max_length,
                jnp.where(done.any(), jnp.where(done, done_lengths, 0).max(), 0),
            ).astype(jnp.float32),
        )

    @staticmethod
    def _empty_episode_metrics() -> EpisodeMetrics:
        """Create initial empty episode metrics."""
        return EpisodeMetrics(
            completed=jnp.array(0),
            sum_returns=jnp.array(0.0),
            sum_lengths=jnp.array(0),
            sum_reward_per_step=jnp.array(0.0),
            min_return=jnp.array(jnp.inf),
            max_return=jnp.array(-jnp.inf),
            min_length=jnp.array(jnp.iinfo(jnp.int32).max, dtype=jnp.float32),
            max_length=jnp.array(0, dtype=jnp.float32),
        )

    # === Instance methods ===

    def init_state(self, seed: int | None = None) -> PPOState:
        """Initialize PPO training state."""
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        key = jax.random.PRNGKey(seed)
        key, param_key, reset_key = jax.random.split(key, 3)

        # Initialize model
        dummy_obs = jnp.zeros((1, self.obs_dim))
        params: Params = dict(self.model.init(param_key, dummy_obs))  # type: ignore[arg-type]
        opt_state = self.optimizer.init(params)

        # Reset environments
        reset_keys = jax.random.split(reset_key, self.config.num_envs)
        obs, env_state = self._reset_fn(reset_keys, self.env_params)

        return PPOState(
            params=params,
            opt_state=opt_state,
            env_state=env_state,
            obs=obs,
            episode_returns=jnp.zeros(self.config.num_envs),
            episode_lengths=jnp.zeros(self.config.num_envs, dtype=jnp.int32),
            rng_key=key,
        )

    def _collect_rollout(
        self,
        state: PPOState,
    ) -> tuple[Rollout, jax.Array, PPOState, EpisodeMetrics]:
        """Collect num_steps of experience from environments."""
        cfg = self.config
        hp = self.hparams

        def step_fn(
            carry: tuple[PPOState, EpisodeMetrics],
            _: None,
        ) -> tuple[
            tuple[PPOState, EpisodeMetrics],
            tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
        ]:
            state, metrics = carry
            key, action_key, step_key = jax.random.split(state.rng_key, 3)

            # Get action and value
            action, log_prob, _entropy, value = self.model.get_action_and_value(
                state.params, state.obs, action_key
            )

            # Step environments
            step_keys = jax.random.split(step_key, cfg.num_envs)
            next_obs, new_env_state, reward, done, _ = self._step_fn(
                step_keys, state.env_state, action, self.env_params
            )

            # Update episode tracking
            new_returns = state.episode_returns + reward
            new_lengths = state.episode_lengths + 1

            # Update metrics for completed episodes
            metrics = PPO._update_episode_metrics(metrics, done, new_returns, new_lengths)

            # Reset counters for done episodes
            new_state = state.replace(  # type: ignore[attr-defined]
                env_state=new_env_state,
                obs=next_obs,
                episode_returns=jnp.where(done, 0.0, new_returns),
                episode_lengths=jnp.where(done, 0, new_lengths),
                rng_key=key,
            )

            transition = (state.obs, action, log_prob, value, reward, done)
            return (new_state, metrics), transition

        init_metrics = PPO._empty_episode_metrics()
        (final_state, episode_metrics), transitions = jax.lax.scan(
            step_fn, (state, init_metrics), None, length=cfg.num_steps
        )

        obs, actions, log_probs, values, rewards, dones = transitions
        rollout = Rollout(
            obs=obs,
            actions=actions,
            log_probs=log_probs,
            values=values,
            rewards=rewards,
            dones=dones,
        )

        # Bootstrap value for GAE
        bootstrap_value = self.model.get_value(final_state.params, final_state.obs)

        # Finalize metrics (convert inf to nan for empty case)
        has_episodes = episode_metrics.completed > 0
        episode_metrics = episode_metrics.replace(  # type: ignore[attr-defined]
            min_return=jnp.where(has_episodes, episode_metrics.min_return, jnp.nan),
            max_return=jnp.where(has_episodes, episode_metrics.max_return, jnp.nan),
            min_length=jnp.where(has_episodes, episode_metrics.min_length, jnp.nan),
            max_length=jnp.where(has_episodes, episode_metrics.max_length, jnp.nan),
        )

        return rollout, bootstrap_value, final_state, episode_metrics

    def _update(
        self,
        params: Params,
        opt_state: optax.OptState,
        rollout: Rollout,
        advantages: jax.Array,
        returns: jax.Array,
        key: jax.Array,
    ) -> tuple[Params, optax.OptState, dict[str, jax.Array]]:
        """Perform PPO update epochs on collected rollout."""
        cfg = self.config
        hp = self.hparams
        batch_size = cfg.batch_size

        # Flatten rollout
        flat_obs = rollout.obs.reshape(batch_size, -1)
        flat_actions = rollout.actions.reshape(batch_size)
        flat_log_probs = rollout.log_probs.reshape(batch_size)
        flat_values = rollout.values.reshape(batch_size)
        flat_advantages = advantages.reshape(batch_size)
        flat_returns = returns.reshape(batch_size)

        def minibatch_step(
            carry: tuple[Params, optax.OptState, dict[str, jax.Array]],
            indices: jax.Array,
        ) -> tuple[tuple[Params, optax.OptState, dict[str, jax.Array]], None]:
            params, opt_state, metrics_acc = carry

            # Extract minibatch
            mb_obs = flat_obs[indices]
            mb_actions = flat_actions[indices]
            mb_log_probs = flat_log_probs[indices]
            mb_advantages = flat_advantages[indices]
            mb_returns = flat_returns[indices]
            mb_values = flat_values[indices]

            loss_fn = lambda p: PPO._ppo_loss(
                p, self.model, mb_obs, mb_actions, mb_log_probs,
                mb_advantages, mb_returns, mb_values,
                hp.clip_coef, hp.vf_coef, hp.ent_coef, hp.clip_vloss,
            )
            grads, metrics = jax.grad(loss_fn, has_aux=True)(params)
            updates, opt_state = self.optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)  # type: ignore[assignment]

            # Accumulate metrics
            new_acc = {k: metrics_acc[k] + metrics[k] for k in metrics_acc}
            return (params, opt_state, new_acc), None  # type: ignore[return-value]

        def epoch_step(
            carry: tuple[Params, optax.OptState, dict[str, jax.Array]],
            key: jax.Array,
        ) -> tuple[tuple[Params, optax.OptState, dict[str, jax.Array]], None]:
            params, opt_state, metrics_acc = carry

            indices = jax.random.permutation(key, batch_size)
            minibatch_indices = indices.reshape(cfg.num_minibatches, cfg.minibatch_size)

            (params, opt_state, metrics_acc), _ = jax.lax.scan(
                minibatch_step, (params, opt_state, metrics_acc), minibatch_indices
            )
            return (params, opt_state, metrics_acc), None

        init_metrics: dict[str, jax.Array] = {
            "policy_loss": jnp.array(0.0),
            "value_loss": jnp.array(0.0),
            "entropy": jnp.array(0.0),
            "total_loss": jnp.array(0.0),
            "approx_kl": jnp.array(0.0),
            "clip_frac": jnp.array(0.0),
        }

        epoch_keys = jax.random.split(key, hp.update_epochs)
        (params, opt_state, final_metrics), _ = jax.lax.scan(
            epoch_step, (params, opt_state, init_metrics), epoch_keys
        )

        # Average metrics
        num_updates = hp.update_epochs * cfg.num_minibatches
        final_metrics = {k: v / num_updates for k, v in final_metrics.items()}

        # Explained variance
        final_metrics["explained_variance"] = 1 - (flat_returns - flat_values).var() / (
            flat_returns.var() + 1e-8
        )

        return params, opt_state, final_metrics

    def _train_step(self, state: PPOState) -> tuple[PPOState, StepMetrics]:
        """Single training step: collect rollout, compute GAE, update policy."""
        hp = self.hparams
        key, update_key = jax.random.split(state.rng_key)

        # Collect rollout
        rollout, bootstrap_value, state, episode_metrics = self._collect_rollout(
            state.replace(rng_key=key)  # type: ignore[attr-defined]
        )

        # Compute GAE
        advantages, returns = PPO._compute_gae(
            rollout.values, rollout.rewards, rollout.dones, bootstrap_value,
            hp.gamma, hp.gae_lambda,
        )

        # Update policy
        params, opt_state, update_metrics = self._update(
            state.params, state.opt_state, rollout, advantages, returns, update_key
        )

        new_state = state.replace(params=params, opt_state=opt_state)  # type: ignore[attr-defined]

        # Compute final episode metrics
        n_completed = episode_metrics.completed
        n_safe = jnp.maximum(n_completed, 1)

        step_metrics = StepMetrics(
            episodes_completed=n_completed,
            avg_return=episode_metrics.sum_returns / n_safe,
            avg_length=episode_metrics.sum_lengths / n_safe,
            avg_reward_per_step=episode_metrics.sum_reward_per_step / n_safe,
            min_return=episode_metrics.min_return,
            max_return=episode_metrics.max_return,
            min_length=episode_metrics.min_length,
            max_length=episode_metrics.max_length,
            policy_loss=update_metrics["policy_loss"],
            value_loss=update_metrics["value_loss"],
            entropy=update_metrics["entropy"],
            total_loss=update_metrics["total_loss"],
            approx_kl=update_metrics["approx_kl"],
            clip_frac=update_metrics["clip_frac"],
            explained_variance=update_metrics["explained_variance"],
        )

        return new_state, step_metrics

    def train_step(self, state: PPOState) -> tuple[PPOState, StepMetrics]:
        """Public JIT-compiled train step."""
        return self._train_step_jit(state)

    def train(self) -> None:
        """Run the full PPO training loop."""
        cfg = self.config
        hp = self.hparams

        # Initialize state
        state = self.init_state(cfg.seed)

        # Log config
        if wandb.run is not None:
            wandb.config.update(asdict(cfg))

        # Warm up JIT
        state, _ = self.train_step(state)
        jax.block_until_ready(state.params)

        start_time = time.perf_counter()
        pbar = tqdm(range(1, cfg.num_iterations), desc="Training")

        for iteration in pbar:
            state, step_metrics = self.train_step(state)
            jax.block_until_ready(state.params)
            step_end = time.perf_counter()

            # Compute metrics
            elapsed = step_end - start_time
            total_steps = (iteration + 1) * cfg.batch_size
            sps = total_steps / elapsed if elapsed > 0 else 0.0

            # Get current LR
            opt_step = state.opt_state[1][1].count  # type: ignore[index]
            lr = optax.linear_schedule(
                init_value=hp.learning_rate,
                end_value=0.0,
                transition_steps=cfg.total_optimizer_steps,
            )(opt_step)

            metrics = {
                "episodes_completed": float(step_metrics.episodes_completed),
                "avg_return": float(step_metrics.avg_return),
                "avg_length": float(step_metrics.avg_length),
                "avg_reward_per_step": float(step_metrics.avg_reward_per_step),
                "min_return": float(step_metrics.min_return),
                "max_return": float(step_metrics.max_return),
                "min_length": float(step_metrics.min_length),
                "max_length": float(step_metrics.max_length),
                "policy_loss": float(step_metrics.policy_loss),
                "value_loss": float(step_metrics.value_loss),
                "entropy": float(step_metrics.entropy),
                "total_loss": float(step_metrics.total_loss),
                "approx_kl": float(step_metrics.approx_kl),
                "clip_frac": float(step_metrics.clip_frac),
                "explained_variance": float(step_metrics.explained_variance),
                "sps": sps,
                "learning_rate": float(lr),
                "time_elapsed": elapsed,
            }

            if wandb.run is not None:
                wandb.log({"iteration": iteration, **metrics}, step=iteration)

            avg_return = metrics["avg_return"]
            return_str = f"{avg_return:.1f}" if not math.isnan(avg_return) else "N/A"
            kl_str = f"{metrics['approx_kl']:.4f}"

            pbar.set_postfix({"SPS": f"{sps:.0f}", "return": return_str, "kl": kl_str})

        pbar.close()

        total_time = time.perf_counter() - start_time
        total_steps = cfg.num_iterations * cfg.batch_size
        print(f"\nTraining completed in {total_time:.1f}s")
        print(f"Total timesteps: {total_steps:,}")
        print(f"Average SPS: {total_steps / total_time:.0f}")
