FROM centos:7

# Install EPEL repository and required build tools.
# openssl11-devel provides OpenSSL 1.1.x headers/libs needed by Python 3.11
# (CentOS 7's default openssl is 1.0.2, which is too old for Python 3.11).
RUN yum -y install epel-release && \
    yum -y update && \
    yum -y install \
        gcc \
        gcc-c++ \
        make \
        wget \
        bzip2 \
        bzip2-devel \
        tar \
        git \
        which \
        zlib-devel \
        libffi-devel \
        xz-devel \
        readline-devel \
        sqlite-devel \
        openssl11 \
        openssl11-devel \
        libX11 \
        libXft \
        libXext \
        libXpm \
        libXtst \
        mesa-libGL \
        mesa-libGLU && \
    yum clean all

# Build Python 3.11 from source.
# SCL only provides up to rh-python39 for CentOS 7, so we compile from source.
# CFLAGS/LDFLAGS direct the build to the openssl11 headers and libraries from EPEL.
ARG PYTHON_VERSION=3.11.9
RUN wget -q "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz" && \
    tar -xzf "Python-${PYTHON_VERSION}.tgz" && \
    cd "Python-${PYTHON_VERSION}" && \
    CFLAGS="-I/usr/include/openssl11" \
    LDFLAGS="-L/usr/lib64/openssl11 -Wl,-rpath,/usr/lib64/openssl11" \
    ./configure --enable-optimizations && \
    make -j$(nproc) altinstall && \
    cd .. && rm -rf "Python-${PYTHON_VERSION}" "Python-${PYTHON_VERSION}.tgz"

# Create default python3 / python / pip3 / pip symlinks
RUN ln -sf /usr/local/bin/python3.11 /usr/local/bin/python3 && \
    ln -sf /usr/local/bin/python3.11 /usr/local/bin/python && \
    ln -sf /usr/local/bin/pip3.11 /usr/local/bin/pip3 && \
    ln -sf /usr/local/bin/pip3.11 /usr/local/bin/pip

# Upgrade pip and install Python scientific packages
# Minimum versions are aligned with requirements.txt in this repository
RUN pip3 install --upgrade pip && \
    pip3 install \
        "pandas>=2.2.2" \
        "numpy>=1.23.5" \
        "awkward>=2.6.6" \
        "uproot>=5.3.10" \
        "matplotlib>=3.4.3" \
        "scikit-image>=0.21.0" \
        "scikit-learn>=1.5.1" \
        "tqdm>=4.66.3" \
        "itables>=2.1.4" \
        skspatial \
        notebook \
        jupyterlab \
        wheel

# Download and install CERN ROOT (latest CentOS 7 compatible binary)
# Update ROOT_VERSION and ROOT_DIST if a newer binary becomes available at:
#   https://root.cern/download/
ARG ROOT_VERSION=6.30.04
ARG ROOT_DIST=Linux-centos7-x86_64-gcc4.8
ARG ROOT_URL=https://root.cern/download/root_v${ROOT_VERSION}.${ROOT_DIST}.tar.gz

RUN mkdir -p /opt && \
    cd /opt && \
    wget -q "${ROOT_URL}" -O root.tar.gz && \
    tar -xzf root.tar.gz && \
    rm root.tar.gz

# Set ROOT environment variables (mirrors what thisroot.sh does)
ENV ROOTSYS=/opt/root
ENV PATH=${ROOTSYS}/bin:${PATH}
ENV LD_LIBRARY_PATH=${ROOTSYS}/lib:${LD_LIBRARY_PATH}
ENV PYTHONPATH=${ROOTSYS}/lib:${PYTHONPATH}

# Also source thisroot.sh for interactive shells (e.g. bash terminals in VSCode)
RUN echo 'source /opt/root/bin/thisroot.sh' >> /etc/bashrc

EXPOSE 8888

WORKDIR /workspaces/SoLAr-V2-reco
