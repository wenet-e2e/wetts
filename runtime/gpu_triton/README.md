### TTS Triton Server


```

# Build the server docker image:
docker build . -f Dockerfile -t tts_server:latest
# start the docker server
docker run --gpus all -v <host path>:<container path> --name tts_server --net host -it tts_server:latest


# export to onnx
python3 vits/export_onnx.py  \
--checkpoint logs/exp/base/G_0.pth \
--cfg configs/base.json \
--onnx_model ./logs/exp/base/generator.onnx \
--providers CUDAExecutionProvider \
--phone data/phones.txt

# model repo preparation
cp generator.onnx model_repo/generator/1/
# please modify the hard coding path in model_repo/tts/config.pbtxt

# start server (inside the container)
CUDA_VISIBLE_DEVICES="0" tritonserver --model-repository model_repo

# start client (inside the container)
python3 client.py --text text.scp --outdir test_audios

# test with triton perf_analyzer tool (inside the docker)
python3 generate_input.py --text text.scp
perf_analyzer -m tts -b 1 -a -p 20000 --concurrency-range 100:200:50 -i gRPC --input-data=./input.json  -u localhost:8001
```