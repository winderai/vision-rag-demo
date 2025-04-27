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
@click.option("--vision-api-key", type=str, default="")
@click.option("--vision-base-url", type=str, default="http://localhost:8000/v1")
@click.option(
    "--vision-embeddings-model", type=str, default="MrLight/dse-qwen2-2b-mrl-v1"
)
@click.option("--embedding-dimension", type=int, default=1536)
@click.option(
    "--db-connection-string",
    type=str,
    default="postgresql://postgres:postgres@localhost:5432/postgres",
)
def cli(
    ctx: click.Context,
    vision_api_key: str,
    vision_base_url: str,
    vision_embeddings_model: str,
    embedding_dimension: int,
    db_connection_string: str,
):
    ctx.ensure_object(dict)
    ctx.obj["vision_api_key"] = vision_api_key
    ctx.obj["vision_base_url"] = vision_base_url
    ctx.obj["vision_embeddings_model"] = vision_embeddings_model
    ctx.obj["embedding_dimension"] = embedding_dimension
    ctx.obj["db_connection_string"] = db_connection_string


@cli.command()
@click.pass_context
@click.argument("file", type=click.Path(exists=True))
def index(
    ctx: click.Context,
    file: str,
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
        vision_api_key=ctx.obj["vision_api_key"],
        vision_base_url=ctx.obj["vision_base_url"],
        vision_embeddings_model=ctx.obj["vision_embeddings_model"],
        embedding_dimension=ctx.obj["embedding_dimension"],
    )
    indexer = IndexingService(config)
    indexer.index_pdf(file)


@cli.command()
@click.pass_context
@click.argument("query", type=str)
def query(
    ctx: click.Context,
    query: str,
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
        vision_api_key=ctx.obj["vision_api_key"],
        vision_base_url=ctx.obj["vision_base_url"],
        vision_embeddings_model=ctx.obj["vision_embeddings_model"],
        embedding_dimension=ctx.obj["embedding_dimension"],
    )
    indexer = IndexingService(config)
    print(json.dumps(indexer.query(query), indent=2))


@cli.command()
@click.pass_context
@click.argument("query", type=str)
@click.option(
    "--model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct"
)  # Also works with gpt-4o-mini
@click.option(
    "--base-url", type=str, default="http://localhost:8001/v1"
)  # Also works with https://api.openai.com/v1
@click.option("--api-key", type=str, default="")
def chat(
    ctx: click.Context,
    query: str,
    model: str,
    base_url: str,
    api_key: str,
):
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
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
        vision_api_key=ctx.obj["vision_api_key"],
        vision_base_url=ctx.obj["vision_base_url"],
        vision_embeddings_model=ctx.obj["vision_embeddings_model"],
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
    if model not in [m.id for m in models]:
        raise click.UsageError(f"Model {model} does not exist")

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that can answer questions about the provided images.",
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

    # response = client.chat.completions.create(
    #     messages=[
    #         {
    #             "role": "system",
    #             "content": "You are a helpful assistant that can answer questions about the provided images.",
    #         },
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "input_text", "text": query},
    #                 *image_content,
    #             ],
    #         },
    #     ],
    #     model=model,
    # )

    # print(response.output_text)
