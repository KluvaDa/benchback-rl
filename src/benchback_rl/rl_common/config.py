"""PPO configuration classes shared across frameworks."""

from typing import Literal, Any
import dataclasses


@dataclasses.dataclass(frozen=True)
class PPOConfig:
    """Full PPO configuration including structural and derived parameters."""

    framework: Literal["torch", "linen", "nnx"]
    env_name: str
    
    # structural parameters
    num_envs: int  # parallel environments
    num_steps: int  # in a rollout
    num_minibatches: int  # number of minibatches per epoch, to split the batch (num_envs*num_steps) into
    update_epochs: int  # number of epochs per update
    num_iterations: int  # training iterations, number of train_step calls
    
    # framework specific parameters
    nnx_jit_mode: str = ""

    # network parameters
    hidden_dim: int = 64 # hidden dimension of actor-critic network

    # logging parameters
    use_wandb: bool = False
    wandb_project: str = "benchback_rl"

    # hyperparameters
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    adam_eps: float = 1e-5
    adam_betas: tuple[float, float] = (0.9, 0.999)
    
    # reproducibility and benchmarking setup
    seed: int | None = None
    sync_for_timing: bool = False

    @property
    def batch_size(self) -> int:
        """Total batch size per iteration (num_envs * num_steps)."""
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        """Size of each minibatch."""
        return self.batch_size // self.num_minibatches

    @property
    def total_timesteps(self) -> int:
        """Total environment timesteps over training."""
        return self.num_iterations * self.batch_size

    @property
    def total_optimizer_steps(self) -> int:
        """Total number of optimizer steps over training."""
        return self.num_iterations * self.update_epochs * self.num_minibatches
    
    def to_dict(self) -> dict[str, Any]:
        """Convert the PPOConfig to a dictionary, including the properties."""
        data_dict = dataclasses.asdict(self)
        data_dict["batch_size"] = self.batch_size
        data_dict["minibatch_size"] = self.minibatch_size
        data_dict["total_timesteps"] = self.total_timesteps
        data_dict["total_optimizer_steps"] = self.total_optimizer_steps
        return data_dict
