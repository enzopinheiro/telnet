FROM python:3.10
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y wget python3-pip
# RUN wget "https://drive.usercontent.google.com/download?id=1BiRyLtkSC-cvuDdOCaet0ZarnFH0ZLJI&export=download&authuser=0&confirm=t&uuid=ec22a9af-7bea-47c6-9000-6ba5c344e24e&at=AN_67v2aaLNPGOqMwUqpeqTKw3Vu:1728010326133" -O /data.tar.gz
#COPY data.tar.gz /data.tar.gz
#RUN tar -xvf /data.tar.gz -C /
#RUN rm /data.tar.gz

COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt
#COPY ../v1p0/ /app/

RUN echo 'alias python="python -W ignore"' >> ~/.bashrc

SHELL ["run", "-n", "/bin/bash", "-c"]
