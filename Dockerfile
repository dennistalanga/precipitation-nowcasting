# Use the official miniconda base image
FROM continuumio/miniconda3:latest

# Define a clean workspace folder inside the virtual container
WORKDIR /workspace

# Install system dependencies required for OpenCV matrix resizing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the environment configuration first to exploit Docker layer caching
COPY environment.yml .

# Build the exact conda environment layer natively inside the machine
RUN conda env create -f environment.yml --quiet && conda clean -afy

# Dynamically force the shell container to execute inside the weather-ml conda context
ENV PATH /opt/conda/envs/weather-ml/bin:$PATH

# Copy the core application and newly built serving deployment folders
COPY src/ ./src
COPY deployment/ ./deployment

# Expose the FastAPI network communication port to the outside world
EXPOSE 8000

# Fire up uvicorn to serve the app on all available interfaces when the container awakens
CMD ["uvicorn", "deployment.app:app", "--host", "0.0.0.0", "--port", "8000"]
