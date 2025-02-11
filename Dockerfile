# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# Configure environment variables
# ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Install system dependencies in a single layer + ffmpeg
RUN apt-get update --fix-missing --no-install-recommends && \
    apt-get install -y \
    curl \
    git \
    ffmpeg \ 
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Set working directory
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
# Add project files individually for better layer caching
COPY README.md ./README.md
COPY pyproject.toml ./pyproject.toml
COPY src/ ./src/

# Install project in editable mode with support for private pip configs
# RUN --mount=type=secret,id=pip,target=/etc/pip.conf \
#     uv pip install -e .

RUN uv pip install  .

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"


# Reset the entrypoint, don't invoke `uv`
ENTRYPOINT []
# Set the info to be default for now 


CMD ["python", "-m", "resoul"]
