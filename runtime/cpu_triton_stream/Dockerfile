FROM jackiexiao/tritonserver:23.10-onnx-py-cpu
# FROM nvcr.io/nvidia/tritonserver:23.10-py3
# https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html
# Here we use a custom-built image instead of the official build for minimal image size

RUN pip3 config set global.index-url https://mirrors.cloud.tencent.com/pypi/simple && \
    pip3 install --no-cache-dir pynini==2.1.5 pypinyin WeTextProcessing

# if you want to include your own model, uncomment the following line
# COPY ./model_repo /models

ENV PYTHONIOENCODING=UTF-8
# 100MB cache
CMD tritonserver --model-repository=/models --cache-config local,size=104857600