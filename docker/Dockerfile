FROM python:3.10
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y wget python3-pip

COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt

RUN echo 'alias python="python -W ignore"' >> ~/.bashrc

SHELL ["run", "-n", "/bin/bash", "-c"]
