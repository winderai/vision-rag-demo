
## Prerequisites

- uv
- Docker
- A machine with a GPU to run the models (tested with a 24GB 4090)

## Setup

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

## Usage

```bash
> uv run vrag --help
Usage: vrag [OPTIONS] COMMAND [ARGS]...

Usage: vrag [OPTIONS] COMMAND [ARGS]...

Options:
  --embedding-dimension INTEGER  [default: 1536]
  --db-connection-string TEXT    [default: postgresql://postgres:postgres@loca
                                 lhost:5432/postgres]
  --help                         Show this message and exit.

Commands:
  chat   Chat with the vector database
  index  Index a PDF file
  query  Query the vector database for results
```

### Index a PDF

```bash
> uv run vrag index --help
Usage: vrag index [OPTIONS] FILE

  Index a PDF file

Options:
  --embedding-api-key TEXT   [default: ""]
  --embedding-base-url TEXT  [default: http://localhost:8000/v1]
  --embedding-model TEXT     [default: MrLight/dse-qwen2-2b-mrl-v1]
  --help                     Show this message and exit.
```

For example:

```bash
uv run vrag index my_file.pdf
```

### Query the Vector DB

```bash
> uv run vrag query --help
Usage: vrag query [OPTIONS] QUERY

  Query the vector database for results

Options:
  --embedding-api-key TEXT   [default: ""]
  --embedding-base-url TEXT  [default: http://localhost:8000/v1]
  --embedding-model TEXT     [default: MrLight/dse-qwen2-2b-mrl-v1]
  --help                     Show this message and exit.
```

For example:

```bash
uv run vrag query "panda" | jq '.[] | [.score, .metadata.file_path]'
```

### Chat With the Vector DB Results

```bash
> uv run vrag chat --help
Usage: vrag chat [OPTIONS] QUERY

  Chat with the vector database

Options:
  --chat-model TEXT          [default: Qwen/Qwen2.5-VL-3B-Instruct]
  --chat-base-url TEXT       [default: http://localhost:8001/v1]
  --chat-api-key TEXT        [default: ""]
  --embedding-api-key TEXT   [default: ""]
  --embedding-base-url TEXT  [default: http://localhost:8000/v1]
  --embedding-model TEXT     [default: MrLight/dse-qwen2-2b-mrl-v1]
  --help                     Show this message and exit.
```

For example:

```bash
uv run vrag chat "what can you see in these images?"
```
