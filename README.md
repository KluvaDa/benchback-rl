# benchback-rl
**Benchmarking Backends for Reinforcement Learning**: Flax.NNX (JAX) vs Flax.Linen (JAX) vs PyTorch

This project aims to compare the three frameworks in pracice, and to evaluate if it's worth starting new RL projects in the new Flax.NNX (Flax version 0.11), or the more mature Flax.Linen or PyTorch.

This repository contains high quality implementation of PPO (Proximal Policy Optimization) that make use of GYMNAX environments that can run on the GPU (Reinforcement Learning Environments in JAX).
 - **PyTorch implementation:**
    - Fully object oriented programming.
    - Uses GYMNAX environments and transfers GPU tensors via DLPack.
- **Linen implementation:**
    - Mostly functional programming, with one PPO class that calls everything.
    - jax.jit compiled functions are big and run once per training iteration.
- **NNX implementation:**
    - Mostly object oriented programming, with scans instead of for loops and special variable tracking that enables nnx.jit.
    - nnx.jit compiled functions that can also be cached using cached_partial.

## Work in progress & TODO
This repository is unfinished. I am actively working on it and it should be done in the next weeks.

### TODO
- [x] Implement PPO in PyTorch
- [x] Implement PPO in Jax, Flax.Linen
- [x] Implement PPO in Jax, Flax.NNX
- [x] Implement entrypoints and benchmarking experiments
- [x] Test and debug everything
- [ ] Finalise Documentation and this readme
- [x] Run all benchmarks
- [ ] Analyse results and present findings in readme

### Current bugs and problems
- The use of different drivers for jax and torch may still be a problem. It doesn't seem to slow anything down, like the transfer of data from jax to torch on the gpu using via DLPack, but vram usage may be affected and competing.

## imporant debugging findings
- flax.nnx cache_partial requires the annotation of static jax arrays or pytrees (specifically encountered with env.env_params) as Variables.
In future version of jax (0.12+) a static annotations exist. In this version (0.11) we need to treat it as a Variable even though its static.
- NNX causes a vram leak when running subsequent benchmarks in the same python script.
jax.clear_caches() mostly addresses the leak, although there may be small leaks left by nnx.cached_partial().
Some memory is also leaked and gc.collect() limits that across all frameworks - this may not be a problem and automatic garbage collection may still work well when running out of memory.

# The rest of this readme is outdated

## Setup

### Environment Variables

This project uses two separate `.env` files:

#### 1. Docker User IDs (`setup/docker/.env`)

Run the setup script to create `setup/docker/.env` with your user/group IDs (required for proper file permissions in containers):

```bash
./setup/scripts/create-env.sh
```

This creates:
```
UID=1000
GID=1000
DOCKER_GID=999
```

#### 2. WandB Credentials (`.env`)

Copy the example file and add your WandB credentials:

```bash
cp .env.example .env
```

Then edit `.env`:
```
WANDB_API_KEY=your_api_key_here
WANDB_ENTITY=your_username_or_team
```

### Running Benchmarks

```bash
python -m benchback_rl.benchmarks.runner
```

## Repository Structure

### Packages and Installation
`setup/docker/Dockerfile.run` with `setup/docker/docker-compose.run.yml` is used to run benchmarks in a reproducible way using `requirements.txt`.

`setup/docker/Dockerfile.dev` with `setup/docker/docker-compose.dev.yml` and `.devcontainer/devcontainer.json` is used for development, installing dependencies from `pyproject.toml`. `setup/scripts/export_requirements.sh` is used to generate `requirements.txt` from within the development container.

This repo installs jax from the docker nvcr.io/nvidia/jax:25.10-py3 container for GPU support. It also installs torch with its bundled CUDA dependencies. This way each package is using its own CUDA libraries for best performance and compatibility at the cost of a larger container image.

### Implementations

All implementations follow the 13 core implementation details from [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/).

#### Design Decisions

**Rollout Buffer Storage Layout**

The buffer stores transitions with the following semantics:
- `obs[t]` — observation fed to the network at step t
- `action[t]`, `log_prob[t]`, `value[t]` — network outputs given `obs[t]`
- `reward[t]`, `done[t]` — **result** of taking `action[t]` in the environment
- `obs[t+1]` — next observation (stored at next index)

This means `done[t]` indicates whether the episode ended *after* taking `action[t]`, not whether `obs[t]` is the first observation of a new episode. The buffer stores `num_steps + 1` observations (including the final bootstrap observation) but only `num_steps` of everything else.

**Termination vs Truncation**

Gymnax environments combine true terminations (agent reached terminal state) and truncations (time limit reached) into a single `done` flag. We accept this simplification, which introduces a small bias for truncated episodes: when an episode is truncated due to time limit, the bootstrap value should ideally be `V(final_obs)` rather than 0, since the episode could have continued. However:
1. For environments with natural termination conditions (CartPole, Atari), true terminations dominate
2. The bias is typically small for well-tuned time limits
3. Handling truncation separately would require modifications to gymnax or manual time tracking

**Buffer Reset Behavior**

The buffer does NOT automatically carry forward the final observation to the next rollout. The caller must explicitly:
1. Call `buffer.reset()` to clear the step counter
2. Call `buffer.set_initial_obs(obs)` with the appropriate starting observation

This explicit API prevents subtle bugs where stale observations might be used.

#### PyTorch
Located in `src/benchback_rl/rl_torch/`, this RL implementation uses PyTorch with an object oriented design. The main training loop is in `train.py`, while the model definitions are in `models.py`. It uses environments that are running on the GPU via `gymnax` using JAX, transferring tensors between PyTorch and JAX using DLPack for efficiency.
#### JAX (Flax.NNX)
Located in `src/benchback_rl/rl_jax_nnx/`, this RL implementation uses JAX with the Flax.NNX library. The design is object oriented, similar to the PyTorch implementation, while allowing jittable jax exectution under the hood, as per Flax.NNX's design philosophy. The main training loop is in `train.py`, while the model definitions are in `models.py`.
#### JAX (Flax.Linen)
Located in `src/benchback_rl/rl_jax_linen/`, this RL implementation uses JAX with the Flax.Linen library. The design is functional, following Flax.Linen's design philosophy. The main training loop is in `train.py`, while the model definitions are in `models.py`.

