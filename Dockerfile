# Use a stable Debian-based Python image
FROM python:3.12-slim-bookworm

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Configure uv to use a virtual environment outside /app to isolate from host's .venv
ENV UV_PROJECT_ENVIRONMENT=/venv
ENV PATH="/venv/bin:$PATH"


WORKDIR /app

# Install system dependencies
# build-essential and python3-dev are required to compile C extensions (e.g. webrtcvad)
# libsndfile1 is required by soundfile at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libsndfile1 \
    libpq5 \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN pip install uv
RUN uv sync --frozen --no-dev

# Copy the rest of the code
COPY . .

# Command to run entrypoint script
CMD ["bash", "entrypoint.sh"]