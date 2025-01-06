FROM ubuntu:22.04

# TODO: everything in docker
# update
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    cmake \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libfontconfig1 libxext6\
    wget
    
# download anaconda3
RUN wget https://repo.anaconda.com/archive/Anaconda3-2023.07-1-Linux-x86_64.sh -O ~/anaconda.sh \
    && /bin/bash ~/anaconda.sh -b -p /opt/conda \
    && rm ~/anaconda.sh

# setup env
ENV PATH=/opt/conda/bin:$PATH
RUN /opt/conda/bin/conda init bash
CMD ["/bin/bash", "-c", "source ~/.bashrc && exec bash"]

# init workspace
WORKDIR /metaurban

# copy all codes
COPY . .

# install requirements
RUN conda create -n metaurban python=3.9
# activate metaurban
RUN /bin/bash -c "source activate metaurban"

# activate metaurban with starting point
RUN echo "source activate metaurban" >> ~/.bashrc

# install dep lib
RUN /opt/conda/envs/metaurban/bin/pip install -e .
RUN /opt/conda/envs/metaurban/bin/pip install scikit-image stable_baselines3 pyyaml
