# benchback-rl
**Benchmarking Backends for Reinforcement Learning**: PyTorch vs JAX (Flax.NNX) vs JAX (Flax.Linen)

## Repo Structure

This repo lists its requirements in `pyproject.toml`, while `requirements.txt` contains pinned exact versions for reproducibility.

`docker/Dockerfile.run` and `docker/Dockerfile.dev` are provided to create reproducible environments for running benchmarks and development, respectively.

- **`docker/Dockerfile.run`**: Used for running benchmarks in a reproducible way using `requirements.txt`
- **`docker/Dockerfile.dev`**: Used for development, installs dependencies from `pyproject.toml` including optional ones.
- **`scripts/export_requirements.sh`**: Generates `requirements.txt` from the existing environment minus the optional dependencies in `pyproject.toml`. It is meant to be executed from within the `docker/Dockerfile.dev` container
- **`.devcontainer.json` and `docker/docker-compose.dev.yml`**: Set up to run the `docker/Dockerfile.dev` container as a VSCode Dev Container
- **`docker/.env`**: Specifies the correct user UID/GID for file permission consistency between host and container