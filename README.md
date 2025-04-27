
## Prerequisites

- Docker
- A machine with a GPU to run the models (tested with a 24GB 4090)

## Usage

1. Start a pgvector container:

```bash
docker run -d -p 5432:5432 -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=postgres ankane/pgvector
```

2. Start a vision embedding model:

```bash
export HUGGING_FACE_HUB_TOKEN=xxxx
docker run -d --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model MrLight/dse-qwen2-2b-mrl-v1 --task embed --max-model-len 8192 --chat-template examples/template_dse_qwen2_vl.jinja
```

3. Start a multimodal language model:

```bash
export HUGGING_FACE_HUB_TOKEN=xxxx
docker run -d --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN" \
    -p 8001:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen2.5-VL-3B-Instruct --max-model-len 32768 --limit-mm-per-prompt image=10
```
