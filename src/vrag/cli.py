import base64
import json
import os

import click
from dotenv import dotenv_values
from haystack.utils import Secret
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from openai import OpenAI

from vrag.service import IndexingService, PDFIndexingServiceConfig

env_vars = dict(dotenv_values())
os.environ.update(env_vars)


@click.group(context_settings=dict(auto_envvar_prefix="VRAG", show_default=True))
@click.pass_context
@click.option("--embedding-dimension", type=int, default=1536)
@click.option(
    "--db-connection-string",
    type=str,
    default="postgresql://postgres:postgres@localhost:5432/postgres",
)
def cli(
    ctx: click.Context,
    embedding_dimension: int,
    db_connection_string: str,
):
    ctx.ensure_object(dict)
    ctx.obj["embedding_dimension"] = embedding_dimension
    ctx.obj["db_connection_string"] = db_connection_string


@cli.command(help="Index a PDF file")
@click.pass_context
@click.argument("file", type=click.Path(exists=True))
@click.option("--embedding-api-key", type=str, default="")
@click.option("--embedding-base-url", type=str, default="http://localhost:8000/v1")
@click.option("--embedding-model", type=str, default="MrLight/dse-qwen2-2b-mrl-v1")
def index(
    ctx: click.Context,
    file: str,
    embedding_api_key: str,
    embedding_base_url: str,
    embedding_model: str,
):
    document_store = PgvectorDocumentStore(
        connection_string=Secret.from_token(ctx.obj["db_connection_string"]),
        embedding_dimension=ctx.obj["embedding_dimension"],
        vector_function="cosine_similarity",
        recreate_table=False,
        search_strategy="hnsw",
        hnsw_recreate_index_if_exists=False,
    )

    config = PDFIndexingServiceConfig(
        document_store=document_store,
        embedding_api_key=embedding_api_key,
        embedding_base_url=embedding_base_url,
        embedding_model=embedding_model,
        embedding_dimension=ctx.obj["embedding_dimension"],
    )
    indexer = IndexingService(config)
    indexer.index_pdf(file)


@cli.command(help="Query the vector database for results")
@click.pass_context
@click.argument("query", type=str)
@click.option("--embedding-api-key", type=str, default="")
@click.option("--embedding-base-url", type=str, default="http://localhost:8000/v1")
@click.option("--embedding-model", type=str, default="MrLight/dse-qwen2-2b-mrl-v1")
def query(
    ctx: click.Context,
    query: str,
    embedding_api_key: str,
    embedding_base_url: str,
    embedding_model: str,
):
    document_store = PgvectorDocumentStore(
        connection_string=Secret.from_token(ctx.obj["db_connection_string"]),
        embedding_dimension=ctx.obj["embedding_dimension"],
        vector_function="cosine_similarity",
        recreate_table=False,
        search_strategy="hnsw",
        hnsw_recreate_index_if_exists=False,
    )

    config = PDFIndexingServiceConfig(
        document_store=document_store,
        embedding_api_key=embedding_api_key,
        embedding_base_url=embedding_base_url,
        embedding_model=embedding_model,
        embedding_dimension=ctx.obj["embedding_dimension"],
    )
    indexer = IndexingService(config)
    print(json.dumps(indexer.query(query), indent=2))


@cli.command(help="Chat with the vector database")
@click.pass_context
@click.argument("query", type=str)
@click.option(
    "--chat-model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct"
)  # Also works with gpt-4o-mini
@click.option(
    "--chat-base-url", type=str, default="http://localhost:8001/v1"
)  # Also works with https://api.openai.com/v1
@click.option("--chat-api-key", type=str, default="vllm-requires-dummy-token")
@click.option("--embedding-api-key", type=str, default="vllm-requires-dummy-token")
@click.option("--embedding-base-url", type=str, default="http://localhost:8000/v1")
@click.option("--embedding-model", type=str, default="MrLight/dse-qwen2-2b-mrl-v1")
def chat(
    ctx: click.Context,
    query: str,
    chat_model: str,
    chat_base_url: str,
    chat_api_key: str,
    embedding_api_key: str,
    embedding_base_url: str,
    embedding_model: str,
):
    client = OpenAI(
        base_url=chat_base_url,
        api_key=chat_api_key,
    )

    document_store = PgvectorDocumentStore(
        connection_string=Secret.from_token(ctx.obj["db_connection_string"]),
        embedding_dimension=ctx.obj["embedding_dimension"],
        vector_function="cosine_similarity",
        recreate_table=False,
        search_strategy="hnsw",
        hnsw_recreate_index_if_exists=False,
    )

    config = PDFIndexingServiceConfig(
        document_store=document_store,
        embedding_api_key=embedding_api_key,
        embedding_base_url=embedding_base_url,
        embedding_model=embedding_model,
        embedding_dimension=ctx.obj["embedding_dimension"],
    )
    indexer = IndexingService(config)
    images = indexer.query(query)

    # Submit the images to the vision API

    image_content = []
    for image in images:
        with open(image["metadata"]["image_path"], "rb") as image_file:
            b64_image = base64.b64encode(image_file.read()).decode("utf-8")

        image_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64_image}"},
            },
        )

    # Check model exists
    models = client.models.list()
    if chat_model not in [m.id for m in models]:
        raise click.UsageError(f"Model {chat_model} does not exist")

    completion = client.chat.completions.create(
        model=chat_model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that has the ability to answer questions about images.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    *image_content,
                ],
            },
        ],
    )

    print(completion.choices[0].message.content)
