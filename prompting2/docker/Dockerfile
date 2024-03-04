FROM nvidia/cuda:11.5.2-cudnn8-runtime-ubuntu20.04

##################
## Requirements ##
##################
SHELL ["/bin/bash", "-c"]
RUN apt-get update
RUN apt-get install -y \
        git \
        python3-pip \
	nano \
    && rm -rf /var/lib/apt/lists/*
RUN export LD_LIBRARY_PATH=/usr/local/cuda/lib64
RUN pip install \
    torch \
    sentencepiece \
    transformers \
    protobuf==3.20.* \
    accelerate \
    nltk \
    datasets \
    evaluate \
    scikit-learn \
    sacrebleu \
    mosestokenizer
RUN python3 -m nltk.downloader punkt
