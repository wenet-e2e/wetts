# Streaming TTS CPU Triton Server

## Quick Start
Run with a pre-built demo
- The VITS model in the Docker image is trained on the Baker dataset with configs/vits2_vocos_v1.json.
- It only trains for 200,000 steps just for demonstration purposes.

```
# server
docker run -d -p8000:8000 -p8001:8001 jackiexiao/baker_tts_server:latest

# stream client
pip install -r requirements-client.txt
cd client/ && python stream_tts_client.py --text text.scp --outdir test_audios
```

You will get the following results:
- Different CPUs may have varying performances. The following results are just for reference.
- CPU(1 core): Intel(R) Xeon(R) Platinum 8255C CPU @ 2.50GHz

```
cd client/ && python3 stream_client.py --text text.scp --outdir test_audios

2|今天天气不好我们家里躺平吧
chunk_id=0, chunk_latency=0.21, chunk_duration=0.75s
chunk_id=1, chunk_latency=0.08, chunk_duration=0.75s
chunk_id=2, chunk_latency=0.04, chunk_duration=0.75s
chunk_id=3, chunk_latency=0.08, chunk_duration=0.75s
chunk_id=4, chunk_latency=0.08, chunk_duration=0.22s
dur=3.21, rtf=0.15, first_latency=0.211
```

## Usage / Commands

See Makefile for details. For example
```sh
make build_docker
make cp_asset
make start_server
make stream_client
make client
```

You need to train and export streaming model first, for example, go to wetts/examples/baker and run
```
# train
bash run.sh --stage 0 --stop_stage 1
# export streaming model
bash run.sh --stage 4 --stop_stage 4
```

## PS
- I enable response_cache in model_repo/tts, if you want to disable it, you can comment out `response_cache` in model_repo/tts/config.pbtxt
- CPU only triton server image: `jackiexiao/tritonserver:23.10-onnx-py-cpu` is built from source code of triton server, which is only 337.83 MB (COMPRESSED SIZE). See below for details.

```
git clone https://github.com/triton-inference-server/server

version=23.10
git checkout r${version}
python3 build.py  \
--enable-logging --enable-stats --enable-tracing --enable-metrics --enable-cpu-metrics \
--cache=local --cache=redis \
--endpoint=http --endpoint=grpc \
--backend=ensemble \
--backend=python \
--backend=onnxruntime

docker tag tritonserver:latest tritonserver:${version}-onnx-py-cpu
```

## Reference
https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html