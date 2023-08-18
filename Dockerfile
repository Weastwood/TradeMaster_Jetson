FROM dustynv/pytorch:1.12-r35.4.1

ENV DEBIAN_FRONTEND=noninteractive

# install prerequisites
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  libopenblas-dev \
		  libopenmpi-dev \
            openmpi-bin \
            openmpi-common \
		  gfortran \
		  libomp-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# download and install the PyTorch wheel
#ARG PYTORCH_URL
#ARG PYTORCH_WHL

#RUN cd /opt && \
#    wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate ${PYTORCH_URL} -O ${PYTORCH_WHL} && \
#    pip3 install --verbose ${PYTORCH_WHL}

RUN python3 -c 'import torch; print(f"PyTorch version: {torch.__version__}"); print(f"CUDA available:  {torch.cuda.is_available()}"); print(f"cuDNN version:   {torch.backends.cudnn.version()}"); print(torch.__config__.show());'

# patch for https://github.com/pytorch/pytorch/issues/45323
RUN PYTHON_ROOT=`pip3 show torch | grep Location: | cut -d' ' -f2` && \
    TORCH_CMAKE_CONFIG=$PYTHON_ROOT/torch/share/cmake/Torch/TorchConfig.cmake && \
    echo "patching _GLIBCXX_USE_CXX11_ABI in ${TORCH_CMAKE_CONFIG}" && \
    sed -i 's/  set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=")/  set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")/g' ${TORCH_CMAKE_CONFIG}

# PyTorch C++ extensions frequently use ninja parallel builds
RUN pip3 install --no-cache-dir scikit-build && \
    pip3 install --no-cache-dir ninja
    
# set the torch hub model cache directory to mounted /data volume
ENV TORCH_HOME=/data/models/torch

RUN wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate https://developer.download.nvidia.com/compute/redist/jp/v512/tensorflow/ -O tensorflow-2.12.0+nv23.06-cp38-cp38-linux_aarch64.whl && \
    pip3 install --no-cache-dir --verbose tensorflow-2.12.0+nv23.06-cp38-cp38-linux_aarch64.whl && \
    rm tensorflow-2.12.0+nv23.06-cp38-cp38-linux_aarch64.whl

# Install Anaconda and dependencies
#RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
#RUN chmod +x ~/miniconda.sh && \
#     ~/miniconda.sh -b -p /opt/conda && \
#     rm ~/miniconda.sh && \
#     /opt/conda/bin/conda create --name TradeMaster python=3.9

#ENV PATH /opt/conda/bin:$PATH

RUN git clone https://github.com/Weastwood/TradeMaster_Jetson.git /home/TradeMaster
RUN cd /home/TradeMaster
#RUN conda update -y conda
#RUN conda init bash
#RUN echo "conda activate TradeMaster" >> ~/.bashrc
#RUN . ~/.bashrc

# Install torch
#RUN /opt/conda/envs/TradeMaster/bin/python -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# Install apex
WORKDIR /home
RUN pip3 install packaging
RUN git clone https://github.com/NVIDIA/apex
WORKDIR apex
RUN pip3 install -v --disable-pip-version-check --no-build-isolation --no-cache-dir ./

# Install requirements
WORKDIR /home/TradeMaster
RUN git pull
RUN pip3 install -r requirements.txt
