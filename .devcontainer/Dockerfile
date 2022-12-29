FROM nvcr.io/nvdlfwea/jax/jax:22.10-py3
#FROM nvdlfwea/jax/jax

RUN apt update && apt install python3-pip -y
RUN python3 -m  pip install torch torchvision torchaudio
RUN python3 -m  pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
RUN python3 -m  pip install 'monai[nibabel, skimage, pillow, tensorboard, gdown, ignite, torchvision, itk, tqdm, lmdb, psutil, cucim, pandas, einops, transformers, mlflow, matplotlib, tensorboardX, tifffile, imagecodecs]'
RUN python3 -m  pip install \
    black \
    pandas \
    numpy \
    comet-ml \
    h5py \
    SimpleITK \
    batchgenerators \
    dask \
    # intensity-normalization[ants] \
    warp-lang \
    report_guided_annotation \
    ott-jax \ 
    evojax[extra] \
    jraph \
    flax \
    toolz \
    ipyparallel \
    optuna \
    equinox \
    einops



RUN git clone https://github.com/SuperElastix/SimpleElastix ${HOME}/externalRepos/elastix

# RUN ["chmod", "+x", "/home/sliceruser/externalRepos/elastix"]
#from https://installati.one/ubuntu/22.04/libpangoxft-1.0-0/
RUN apt update
RUN apt -y install libtbb-dev
#from https://askubuntu.com/questions/409310/error-while-loading-shared-libraries-cannot-open-shared-object-file-no-such-fi
RUN apt -y install libgtk-3-dev
RUN apt update

RUN apt-get -y install libidn11-dev  \
                libglu1-mesa-dev  \
                libpangox-1.0-dev  \
                libpangoxft-1.0-0

RUN mkdir ${HOME}/elastixBase && \
cd ${HOME}/elastixBase && \
wget https://github.com/SuperElastix/elastix/releases/download/5.0.1/elastix-5.0.1-linux.tar.bz2&& \
tar -xf elastix-5.0.1-linux.tar.bz2

RUN cp ${HOME}/elastixBase/elastix-5.0.1-linux/lib/libANNlib-5.0.so ${HOME}/elastixBase/elastix-5.0.1-linux/bin/libANNlib-5.0.so
RUN cp ${HOME}/elastixBase/elastix-5.0.1-linux/lib/libANNlib-5.0.so.1 ${HOME}/elastixBase/elastix-5.0.1-linux/bin/libANNlib-5.0.so.1
RUN cp ${HOME}/elastixBase/elastix-5.0.1-linux/lib/libANNlib-5.0.so ${HOME}/elastixBase/elastix-5.0.1-linux/libANNlib-5.0.so
RUN cp ${HOME}/elastixBase/elastix-5.0.1-linux/lib/libANNlib-5.0.so.1 ${HOME}/elastixBase/elastix-5.0.1-linux/libANNlib-5.0.so.1
RUN cp ${HOME}/elastixBase/elastix-5.0.1-linux/lib/libANNlib-5.0.so /usr/local/lib/libANNlib-5.0.so
RUN cp ${HOME}/elastixBase/elastix-5.0.1-linux/lib/libANNlib-5.0.so.1 /usr/local/lib/libANNlib-5.0.so.1

RUN ldconfig
RUN apt-get install -q -y vim nano 

#install go for jax memory profiling


ENV GO_VERSION=1.19.4

RUN apt-get update
RUN apt-get install -y wget git gcc

RUN wget -P /tmp "https://dl.google.com/go/go${GO_VERSION}.linux-amd64.tar.gz"

RUN tar -C /usr/local -xzf "/tmp/go${GO_VERSION}.linux-amd64.tar.gz"
RUN rm "/tmp/go${GO_VERSION}.linux-amd64.tar.gz"

ENV GOPATH /go
ENV PATH $GOPATH/bin:/usr/local/go/bin:$PATH
RUN mkdir -p "$GOPATH/src" "$GOPATH/bin" && chmod -R 777 "$GOPATH"

ENV GOPATH /go
ENV PATH $GOPATH/bin:/usr/local/go/bin:$PATH
RUN mkdir -p "$GOPATH/src" "$GOPATH/bin" && chmod -R 777 "$GOPATH"


RUN apt install graphviz
RUN go install github.com/google/pprof@latest

# # install python3-pip
# RUN apt update && apt install python3-pip -y

# RUN pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

