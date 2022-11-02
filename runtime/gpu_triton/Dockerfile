FROM nvcr.io/nvidia/tritonserver:22.09-py3
# https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html
# Please choose previous tritonserver:xx.xx if you encounter cuda driver mismatch issue

LABEL maintainer="NVIDIA"
LABEL repository="tritonserver"

RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip3 install pypinyin scipy WeTextProcessing grpcio-tools tritonclient
WORKDIR /workspace
