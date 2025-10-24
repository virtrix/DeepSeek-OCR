# FROM python:3.12-bookworm
FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel

WORKDIR /root

# copy files
COPY requirements.txt .
COPY vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl .
COPY server.py .

# install dependencies
# RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
RUN pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
RUN pip install -r requirements.txt

# Install git (required for flash-attn submodule cloning)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN pip install flash-attn==2.7.3 --no-build-isolation

# Clean up wheel file after installation
RUN rm /root/vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl

EXPOSE 7860
CMD ["python", "server.py"]