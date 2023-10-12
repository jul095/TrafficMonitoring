FROM nvidia/cuda:11.0.3-runtime-ubuntu18.04

#################################
## update/ install global dependencies
#################################
RUN apt-get update -y && \
    apt-get upgrade -y


#################################
## create user
#################################
ARG USERNAME="efsfkk"
ARG CONDA_ENV="condaefs"

# Create user folders
RUN mkdir -p /home/${USERNAME} && mkdir -p /home/${USERNAME}/videos

# Create a non-root user
ARG uid=1000
ARG gid=100
ENV USER ${USERNAME}
ENV UID 1000
ENV GID 100
ENV HOME /home/${USERNAME}

RUN adduser --disabled-password \
    --gecos "Non-root user" \
    --uid ${UID} \
    --gid ${GID} \
    --home ${HOME} \
    ${USER}

RUN chown -R ${UID}:${GID} ${HOME}


#################################
## Basic installation
#################################
# This environment variable is nessecary for the apt-installation
ENV DEBIAN_FRONTEND noninteractive

# installation dependencies for miniconda, python and detectron2
RUN apt-get update && \
    apt-get install -y \
    wget \
    python3-pip \
    git \
    python3-opencv \
    ca-certificates \
    python3-dev \
    sudo \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home/${USERNAME}

#################################
## install miniconda
#################################
ENV CONDA_DIR /home/${USERNAME}/miniconda3
ENV ENV_PREFIX ${CONDA_DIR}/env

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p ${CONDA_DIR} && \
    rm -v ~/miniconda.sh

ENV PATH=${CONDA_DIR}/bin:${PATH}
RUN chown -R ${UID}:${GID} ${CONDA_DIR}
RUN chown -R ${UID}:${GID} /home/${USERNAME}/.conda

# Change User
USER ${USERNAME}
WORKDIR /home/${USERNAME}

# Install dependencies
COPY environment.yml ./environment.yml

RUN conda update --name base --channel defaults conda && \
    conda install python=3.8 && \
    conda env create --file environment.yml --name ${CONDA_ENV} --force && \
    conda clean --all --yes

# activate conda environment for bash shell
RUN echo ". ${CONDA_DIR}/etc/profile.d/conda.sh" >> /home/${USERNAME}/.profile
RUN conda init bash


#################################
## install detectron2
#################################
ENV PATH="/home/$USERNAME/.local/bin:${PATH}"

## See https://pytorch.org/ for other options if you use a different version of CUDA
RUN python -m pip install torch==1.7.1 torchvision==0.8.2


#RUN python -m pip install --user 'git+https://github.com/facebookresearch/detectron2.git'
RUN python -m pip install --user detectron2 -f \
    https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
#  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html


#################################
## Install dependencies
#################################
RUN conda init bash && \
    pip install cv2module pandas shapely scipy imantics sklearn

### Upload File - Dependencies ####
RUN pip install azure-storage-blob

#################################
## copy trafficmonitoring -repo
#################################
USER ${USERNAME}
COPY ./ ./trafficmonitoring

########################do#########
## create Entrypoint
################################
WORKDIR /home/${USERNAME}/trafficmonitoring/traffic_monitoring
USER root
#@TODO give user {USERNAME} permission on the directory where we get the video mounted


ENTRYPOINT [ "python", "run_on_video.py", "--video"]
#CMD ["/mnt/trafficmonitoring/small_example_video.mp4"]

#ENTRYPOINT [ "python", "run_on_video.py", "--video", "./videos/small_example_video.mp4" ]


