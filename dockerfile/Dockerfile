FROM nvidia/cuda:11.6.1-devel-ubuntu18.04
USER root
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y \
    build-essential \
    wget \
    git \
    libbz2-dev \
    libdb-dev \
    libreadline-dev \
    libffi-dev \
    libgdbm-dev \
    liblzma-dev \
    libncursesw5-dev \
    libsqlite3-dev \
    libssl-dev \
    zlib1g-dev \
    uuid-dev \
    tk-dev
RUN apt-get install -y --allow-unauthenticated graphviz
RUN apt install -y curl ffmpeg
RUN curl -OL https://www.python.org/ftp/python/3.9.10/Python-3.9.10.tar.xz
RUN tar xJf Python-3.9.10.tar.xz
RUN cd Python-3.9.10 && ./configure && make && make install
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
RUN python3 -m pip install numpy
RUN python3 -m pip install scikit-video
RUN python3 -m pip install opencv-python
RUN python3 -m pip install nvidia-ml-py3
RUN python3 -m pip install lpips
RUN python3 -m pip install scikit-image
RUN python3 -m pip install scikit-learn
RUN python3 -m pip install tqdm
RUN python3 -m pip install matplotlib
RUN python3 -m pip install imgcat
RUN python3 -m pip install pandas
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all
