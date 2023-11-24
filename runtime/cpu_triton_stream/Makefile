# gpu/cpu image
triton_image=nvcr.io/nvidia/tritonserver:23.10-py3
# cpu only image (smaller image size)
triton_image=jackiexiao/tritonserver:23.10-onnx-py-cpu

exp_name=vits2_vocos_v1
model_dir=${shell pwd}/model_repo
repo_dir=${shell dirname ${shell dirname ${shell pwd}}}
ckpt_step=200000


# Build the server docker image:
build_docker:
	docker build . -f Dockerfile -t tts_server:latest

.PHONY: cp_asset
# Copy the asset to the model repo
cp_asset:
	cp ${repo_dir}/examples/baker/exp/${exp_name}/encoder_G_${ckpt_step}.onnx ${model_dir}/encoder/1/encoder.onnx
	cp ${repo_dir}/examples/baker/exp/${exp_name}/decoder_G_${ckpt_step}.onnx ${model_dir}/decoder/1/decoder.onnx
	cp ${repo_dir}/examples/baker/data/lexicon.txt ${model_dir}/lexicon.txt
	cp ${repo_dir}/examples/baker/data/phones.txt ${model_dir}/phones.txt
	cp ${repo_dir}/examples/baker/configs/${exp_name}.json ${model_dir}/base.json

.PHONY: start_server
start_server:
	docker run \
		--rm \
		--cpus 2 \
		-p8000:8000 -p8001:8001 -p8002:8002 \
		--shm-size=1g \
		-v ${model_dir}:/models \
		--name tts_triton_server \
		tts_server:latest \
		bash -c "tritonserver --model-repository=/models --cache-config local,size=104857600"

# streaming client
.PHONY: stream_client
stream_client:
	cd client/ && python stream_client.py --text text.scp --outdir test_audios

# non streaming client
.PHONY: client
client:
	cd client/ && python client.py --text text.scp --outdir test_audios

.PHONY: web_ui
web_ui:
	cd client/ && streamlit run web_ui.py