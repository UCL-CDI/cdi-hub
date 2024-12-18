FROM ubuntu:24.04
FROM python:3.12

# FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND noninteractive

SHELL ["/bin/bash", "-cu"]


RUN apt-get update && apt-get install -y python3

COPY . .
RUN pip install --upgrade pip
RUN pip install --editable .

ENTRYPOINT ["tail", "-f", "/dev/null"]
