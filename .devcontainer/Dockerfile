FROM ubuntu:24.04

# Update package lists and install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    cmake \
    make \
    openssl \
    libssl-dev \
    libbz2-dev \
    libffi-dev \
    zlib1g-dev \
    libx11-dev \
    libxpm-dev \
    libxft-dev \
    libxext-dev \
    libfreetype-dev \
    libpng-dev \
    libjpeg-dev \
    wget \
    bzip2 \
    tar \
    git \
    which \
    bash \
    ca-certificates \
    curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Miniforge for the current architecture (conda-forge based, no Anaconda ToS gate),
# then create a dedicated environment.
RUN ARCH="$(dpkg --print-architecture)" && \
    if [ "$ARCH" = "amd64" ]; then CONDA_ARCH="x86_64"; \
    elif [ "$ARCH" = "arm64" ]; then CONDA_ARCH="aarch64"; \
    else echo "Unsupported architecture: $ARCH" && exit 1; fi && \
    wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${CONDA_ARCH}.sh" -O miniforge.sh && \
    bash miniforge.sh -b -p /opt/conda && \
    rm miniforge.sh && \
    /opt/conda/bin/conda config --system --prepend channels conda-forge && \
    /opt/conda/bin/conda config --system --set channel_priority strict && \
    /opt/conda/bin/conda create -y -n solarv2 -c conda-forge python=3.11 root pip cython setuptools numpy && \
    /opt/conda/bin/conda clean -afy

# Ensure conda is available in all shells and default to the project environment
ENV PATH=/opt/conda/bin:$PATH
RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate solarv2" >> /root/.bashrc

EXPOSE 8888

WORKDIR /workspaces/SoLAr-V2-reco
