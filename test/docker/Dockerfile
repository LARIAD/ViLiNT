FROM nvcr.io/nvidia/isaac-sim:5.1.0

ENV DEBIAN_FRONTEND=noninteractive

USER root
RUN echo 'root:1234' | chpasswd
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    gnupg2 \
    locales \
    lsb-release \
    git \
    nano \
    ninja-build \
    openssh-server \
    wget \
    tmux \
    vim \
&& rm -rf /var/lib/apt/lists/*

# Prepare requirements for ros2 install 
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      software-properties-common \
 && add-apt-repository universe \
 && add-apt-repository multiverse \
 && apt-get update \
 && apt-get install -y --allow-downgrades libbrotli1 libbrotli-dev \
 && apt-get install -y --no-install-recommends \
      libfreetype6-dev \
      libfontconfig1-dev \
      libxft-dev \
      libxrender-dev \
      libglu1-mesa-dev \
      libglew-dev \
      libx11-dev \
      libx11-xcb-dev \
 && apt-get install -f \
 && apt-get update 

# Install ros2
RUN apt-get update && apt-get install -y curl \
 && export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}') \
 && curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb" \
 && dpkg -i /tmp/ros2-apt-source.deb \
 && rm -f /tmp/ros2-apt-source.deb \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      ros-jazzy-desktop \
      ros-dev-tools \
      ros-jazzy-tf-transformations \
 && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-lc"]
#RUN echo "source /opt/ros/jazzy/setup.bash" >> /etc/bash.bashrc

# Install Miniconda 
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}

RUN wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash /tmp/miniconda.sh -b -p "${CONDA_DIR}" \
    && rm -f /tmp/miniconda.sh \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && conda config --system --set auto_activate_base false \
    && conda create -y -n lint python=3.12 \
    && conda clean -afy

RUN echo ". ${CONDA_DIR}/etc/profile.d/conda.sh" >> /etc/bash.bashrc

ARG CUDA_MAJOR=12
ARG CUDA_MINOR=8
ARG ISAACSIM_VERSION=5.1.0

RUN CUDA_DISTRO="$(. /etc/os-release && echo "ubuntu${VERSION_ID//./}")" \
    && wget -qO /tmp/cuda-keyring.deb "https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_DISTRO}/x86_64/cuda-keyring_1.1-1_all.deb" \
    && dpkg -i /tmp/cuda-keyring.deb \
    && rm -f /tmp/cuda-keyring.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends "cuda-toolkit-${CUDA_MAJOR}-${CUDA_MINOR}" \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda-${CUDA_MAJOR}.${CUDA_MINOR}
ENV PATH=${CUDA_HOME}/bin:${CONDA_DIR}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"

WORKDIR /workspace

# Install ViLiNT dependencies
RUN conda run --no-capture-output -n lint pip install --upgrade pip setuptools wheel numpy \
    && conda run --no-capture-output -n lint pip install --upgrade torch torchvision  --index-url https://download.pytorch.org/whl/cu128 \
    && conda run --no-capture-output -n lint pip install --upgrade \
        torchfix \
        timm \
        "huggingface-hub==0.36.0" \
        "transformers==4.57.6" \
        accelerate \
        einops \
        torchmetrics \
        optuna \
        tensorboard \
        matplotlib \
        pandas \
        jaxtyping \
        omegaconf \
        ipython \
        black \
        flake8 \
        pylint \
        rich \
        ipykernel \
    && conda run --no-capture-output -n lint pip install --upgrade \
        h5py \
        pyyaml \
        sharedarray \
        tensorboardx \
        yapf \
        addict \
        scipy \
        plyfile \
        termcolor 
RUN PYG_TORCH="$(${CONDA_DIR}/envs/lint/bin/python -c "import torch; v = torch.__version__.split('+')[0].split('.'); print(f'{v[0]}.{v[1]}.0')")" \
    && PYG_CUDA="$(${CONDA_DIR}/envs/lint/bin/python -c "import torch; print('cu' + torch.version.cuda.replace('.', ''))")" \
    && conda run --no-capture-output -n lint pip install --upgrade torch-geometric \
    && conda run --no-capture-output -n lint pip install --no-cache-dir --no-index \
        --find-links "https://data.pyg.org/whl/torch-${PYG_TORCH}+${PYG_CUDA}.html" \
        pyg_lib \
        torch_cluster \
        torch_scatter \
        torch_sparse \
    && conda run --no-capture-output -n lint pip install --upgrade torch-geometric 

RUN conda run --no-capture-output -n lint pip install --upgrade \
        tqdm \
        opencv-python-headless \
        wandb \
        prettytable \
        efficientnet-pytorch \
        "warmup-scheduler==0.3" \
        "diffusers==0.11.1" \
        lmdb \
        vit-pytorch \
        positional-encodings \
        zarr \
        transforms3d \
    && sed -i 's/, cached_download//g' /opt/conda/envs/lint/lib/python3.12/site-packages/diffusers/dynamic_modules_utils.py

ENV CUMM_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

RUN git clone --depth 1 https://github.com/FindDefinition/cumm.git ~/cumm \
    && cd ~/cumm \
    && conda run --no-capture-output -n lint python -m pip install -e . \
    && conda run --no-capture-output -n lint python -c "import cumm"

ENV CUMM_DISABLE_JIT=1

RUN git clone --depth 1 https://github.com/traveller59/spconv.git ~/spconv \
    && cd ~/spconv \
    && sed -i 's/, <0.8.0//g' setup.py pyproject.toml

RUN cd ~/spconv \
    && conda run --no-capture-output -n lint python -m pip install --no-build-isolation -e . \
    && conda run --no-capture-output -n lint python -c "import torch; import spconv; import spconv.pytorch"

ENV SPCONV_DISABLE_JIT=1

ARG FLASH_ATTN_MAX_JOBS=2
ARG FLASH_ATTN_NVCC_THREADS=2

RUN conda run --no-capture-output -n lint env \
       MAX_JOBS=${FLASH_ATTN_MAX_JOBS} \
       NVCC_THREADS=${FLASH_ATTN_NVCC_THREADS} \
       CMAKE_BUILD_PARALLEL_LEVEL=${FLASH_ATTN_MAX_JOBS} \
       pip install --upgrade --no-cache-dir --no-build-isolation flash-attn 

RUN conda create -y -n isaac python=3.11 \
    && conda run --no-capture-output -n isaac python -m pip install --upgrade pip setuptools wheel \
    && conda run --no-capture-output -n isaac python -m pip install "isaacsim[all,extscache]==${ISAACSIM_VERSION}" --extra-index-url https://pypi.nvidia.com \
    && conda clean -afy

# Temporary fix
RUN conda run --no-capture-output -n lint python -m pip install numpy==1.*

ENV CUDA_HOME=
ENV PATH=${CONDA_DIR}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64


COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
