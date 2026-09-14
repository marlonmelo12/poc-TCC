FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install build tools and libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install remaining Python requirements (PyTorch 2.5.1 with CUDA is already pre-installed!)
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Try installing pyopf if supported
RUN pip install --no-cache-dir pyopf || echo "pyopf installation skipped, fallback to NumPy OPF if needed"

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
