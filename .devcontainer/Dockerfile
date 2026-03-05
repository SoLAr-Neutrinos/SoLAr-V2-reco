FROM centos:7

# Install EPEL and Software Collections (SCL) repositories,
# then install required build tools and Python 3.9
RUN yum -y install epel-release centos-release-scl && \
    yum -y update && \
    yum -y install \
        gcc \
        gcc-c++ \
        make \
        rh-python39 \
        rh-python39-python-devel \
        rh-python39-python-pip \
        wget \
        bzip2 \
        tar \
        git \
        which \
        libX11 \
        libXft \
        libXext \
        libXpm \
        libXtst \
        mesa-libGL \
        mesa-libGLU \
        openssl-devel && \
    yum clean all

# Make Python 3.9 from SCL the default Python environment
ENV PATH=/opt/rh/rh-python39/root/usr/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/rh/rh-python39/root/usr/lib64:${LD_LIBRARY_PATH}
ENV PKG_CONFIG_PATH=/opt/rh/rh-python39/root/usr/lib64/pkgconfig:${PKG_CONFIG_PATH}
ENV X_SCLS=rh-python39

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
