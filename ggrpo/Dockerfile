# Use the official NVIDIA PyTorch base image for GPU-accelerated workloads
FROM nvcr.io/nvidia/pytorch:24.03-py3

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements or dependency files
# Note: Since the project uses uv/pyproject.toml, we can install dependencies directly
RUN pip install --no-cache-dir \
    "transformers>=4.48.0" \
    "torch>=2.2.0" \
    "matplotlib>=3.8.0" \
    "gymnasium>=0.29.0"

# Copy the GRPO source code into the container
COPY dataset.py reward.py training.py /app/

# Set environment variables for CUDA/PyTorch optimization
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
ENV PYTHONUNBUFFERED=1

# Command to execute the training loop
CMD ["python", "training.py"]
