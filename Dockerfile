FROM nvcr.io/nvidia/jax:25.10-py3

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /workspace/benchback-rl

# Enable bytecode compilation for better startup performance
ENV UV_COMPILE_BYTECODE=1

# Copy only the necessary package directories and metadata files
COPY libs/ ./libs/
COPY benchback_runner/ ./benchback_runner/

# Install the benchback_runner package and its dependencies using uv
# --system flag installs to system Python (no venv needed in Docker)
# This will install all three backend libraries
RUN uv pip install --system -e ./benchback_runner

# Placeholder command - will be replaced with actual benchmark runner
CMD ["python", "-c", "print('Benchback RL container ready. Add your command here.')"]
