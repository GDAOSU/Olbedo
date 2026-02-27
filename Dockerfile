FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget \
    git \
    curl \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

ENV PATH=$CONDA_DIR/bin:$PATH

WORKDIR /app

COPY requirements.txt /app/

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -n olbedo_onr python=3.10.12 -y && \
    /bin/bash -c "source activate olbedo_onr && pip install --no-cache-dir -r requirements.txt"

COPY olbedo /app/olbedo
COPY script /app/script

RUN /bin/bash -c "source activate olbedo_onr && python script/download_weights.py"

RUN echo "source activate olbedo_onr" >> ~/.bashrc

CMD ["/bin/bash"]