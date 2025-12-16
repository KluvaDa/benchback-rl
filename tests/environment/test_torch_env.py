"""Tests for TorchEnv wrapper."""

import jax
import jax.numpy as jnp
import pytest
import torch

from benchback_rl.environment.torch_env import (
    TorchEnv,
    jax_pytree_to_torch,
    jax_to_torch,
    torch_to_jax,
)


class TestJaxToTorch:
    def test_converts_jax_array_to_torch_tensor(self) -> None:
        x = jnp.array([1.0, 2.0, 3.0])
        result = jax_to_torch(x)

        assert isinstance(result, torch.Tensor)
        assert result.tolist() == [1.0, 2.0, 3.0]

    def test_result_is_on_cuda(self) -> None:
        x = jnp.array([1.0, 2.0, 3.0])
        result = jax_to_torch(x)

        assert result.device.type == "cuda"

    def test_preserves_shape(self) -> None:
        x = jnp.zeros((4, 3, 2))
        result = jax_to_torch(x)

        assert result.shape == (4, 3, 2)

    def test_preserves_dtype_float32(self) -> None:
        x = jnp.array([1.0], dtype=jnp.float32)
        result = jax_to_torch(x)

        assert result.dtype == torch.float32

    def test_preserves_dtype_int32(self) -> None:
        x = jnp.array([1], dtype=jnp.int32)
        result = jax_to_torch(x)

        assert result.dtype == torch.int32


class TestTorchToJax:
    def test_converts_torch_tensor_to_jax_array(self) -> None:
        x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        result = torch_to_jax(x)

        assert isinstance(result, jax.Array)
        assert result.tolist() == [1.0, 2.0, 3.0]

    def test_preserves_shape(self) -> None:
        x = torch.zeros((4, 3, 2), device="cuda")
        result = torch_to_jax(x)

        assert result.shape == (4, 3, 2)

    def test_preserves_dtype_float32(self) -> None:
        x = torch.tensor([1.0], dtype=torch.float32, device="cuda")
        result = torch_to_jax(x)

        assert result.dtype == jnp.float32

    def test_preserves_dtype_int32(self) -> None:
        x = torch.tensor([1], dtype=torch.int32, device="cuda")
        result = torch_to_jax(x)

        assert result.dtype == jnp.int32


class TestJaxPytreeToTorch:
    def test_converts_dict_of_arrays(self) -> None:
        tree = {"a": jnp.array([1.0]), "b": jnp.array([2.0])}
        result = jax_pytree_to_torch(tree)

        assert isinstance(result["a"], torch.Tensor)
        assert isinstance(result["b"], torch.Tensor)

    def test_converts_nested_dict(self) -> None:
        tree = {"outer": {"inner": jnp.array([1.0])}}
        result = jax_pytree_to_torch(tree)

        assert isinstance(result["outer"]["inner"], torch.Tensor)

    def test_converts_list_of_arrays(self) -> None:
        tree = [jnp.array([1.0]), jnp.array([2.0])]
        result = jax_pytree_to_torch(tree)

        assert isinstance(result[0], torch.Tensor)
        assert isinstance(result[1], torch.Tensor)

    def test_passes_through_non_arrays(self) -> None:
        tree = {"array": jnp.array([1.0]), "string": "hello", "int": 42}
        result = jax_pytree_to_torch(tree)

        assert isinstance(result["array"], torch.Tensor)
        assert result["string"] == "hello"
        assert result["int"] == 42


class TestTorchEnv:
    @pytest.fixture
    def env(self) -> TorchEnv:
        return TorchEnv("CartPole-v1", num_envs=4, jit=True)

    @pytest.fixture
    def key(self) -> jax.Array:
        return jax.random.PRNGKey(0)

    def test_init_sets_num_envs(self, env: TorchEnv) -> None:
        assert env.num_envs == 4

    def test_init_sets_num_actions(self, env: TorchEnv) -> None:
        assert env.num_actions == 2  # CartPole has 2 actions

    def test_init_sets_params(self, env: TorchEnv) -> None:
        assert env.params is not None

    def test_reset_returns_torch_tensor_obs(
        self, env: TorchEnv, key: jax.Array
    ) -> None:
        obs = env.reset(key)

        assert isinstance(obs, torch.Tensor)

    def test_reset_obs_on_cuda(self, env: TorchEnv, key: jax.Array) -> None:
        obs = env.reset(key)

        assert obs.device.type == "cuda"

    def test_reset_obs_shape(self, env: TorchEnv, key: jax.Array) -> None:
        obs = env.reset(key)

        assert obs.shape == (4, 4)  # (num_envs, obs_dim) for CartPole

    def test_reset_sets_internal_state(self, env: TorchEnv, key: jax.Array) -> None:
        env.reset(key)

        assert env._state is not None

    def test_step_returns_torch_tensors(
        self, env: TorchEnv, key: jax.Array
    ) -> None:
        env.reset(key)
        key, step_key = jax.random.split(key)
        actions = torch.zeros(4, dtype=torch.int32, device="cuda")

        obs, reward, done, info = env.step(step_key, actions)

        assert isinstance(obs, torch.Tensor)
        assert isinstance(reward, torch.Tensor)
        assert isinstance(done, torch.Tensor)

    def test_step_tensors_on_cuda(self, env: TorchEnv, key: jax.Array) -> None:
        env.reset(key)
        key, step_key = jax.random.split(key)
        actions = torch.zeros(4, dtype=torch.int32, device="cuda")

        obs, reward, done, info = env.step(step_key, actions)

        assert obs.device.type == "cuda"
        assert reward.device.type == "cuda"
        assert done.device.type == "cuda"

    def test_step_obs_shape(self, env: TorchEnv, key: jax.Array) -> None:
        env.reset(key)
        key, step_key = jax.random.split(key)
        actions = torch.zeros(4, dtype=torch.int32, device="cuda")

        obs, reward, done, info = env.step(step_key, actions)

        assert obs.shape == (4, 4)

    def test_step_reward_shape(self, env: TorchEnv, key: jax.Array) -> None:
        env.reset(key)
        key, step_key = jax.random.split(key)
        actions = torch.zeros(4, dtype=torch.int32, device="cuda")

        obs, reward, done, info = env.step(step_key, actions)

        assert reward.shape == (4,)

    def test_step_done_shape(self, env: TorchEnv, key: jax.Array) -> None:
        env.reset(key)
        key, step_key = jax.random.split(key)
        actions = torch.zeros(4, dtype=torch.int32, device="cuda")

        obs, reward, done, info = env.step(step_key, actions)

        assert done.shape == (4,)

    def test_step_info_contains_discount(
        self, env: TorchEnv, key: jax.Array
    ) -> None:
        env.reset(key)
        key, step_key = jax.random.split(key)
        actions = torch.zeros(4, dtype=torch.int32, device="cuda")

        obs, reward, done, info = env.step(step_key, actions)

        assert "discount" in info
        assert isinstance(info["discount"], torch.Tensor)

    def test_jit_false_still_works(self, key: jax.Array) -> None:
        env = TorchEnv("CartPole-v1", num_envs=2, jit=False)
        obs = env.reset(key)

        assert obs.shape == (2, 4)

    def test_continuous_action_env_raises_error(self) -> None:
        with pytest.raises(ValueError, match="Only Discrete action spaces supported"):
            TorchEnv("Pendulum-v1", num_envs=2, jit=True)
