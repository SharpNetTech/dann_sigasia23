FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04 AS build

# Install git, python
RUN apt-get update && apt-get install -y git python3.10 python3-pip python-is-python3 cmake ffmpeg

# Install dependencies
RUN git clone --recursive "https://github.com/BachiLi/diffvg.git" /tmp/diffvg
COPY requirements.txt /tmp/diffvg/requirements.txt
COPY constraints.txt /tmp/diffvg/constraints.txt
RUN pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128
RUN pip install -r /tmp/diffvg/requirements.txt -c /tmp/diffvg/constraints.txt
RUN cd /tmp/diffvg && python setup.py install

WORKDIR /workspace
