FROM ubuntu:20.04

RUN apt update -y && \
    apt install -y imagemagick wget

RUN mkdir -p /workspace
WORKDIR /workspace

COPY lena.pgm lena.pgm

# CMD /bin/bash