# Repository structure
This repo contains separate libraries for the RL implementations:
- `libs/rl_linen`
- `libs/rl_nnx`
- `libs/rl_torch`

`benchback_runner` contains the main runner code to benchmark different backends, with `Dockerfile` for easy setup.

`Dockerfile.dev` is provided for development purposes and running in VSCode devcontainers. Development packages are specified in `benchback_runner/pyproject.toml` under the `[project.optional-dependencies]` `dev` section.
