from dataclasses import dataclass
from typing import Any, Dict, List
from haystack import Document, Pipeline
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
import logging

from vrag.components import PDFToImagesConverter
from vrag.splitters import ImageSplitter
from vrag.embedders import MultimodalDocumentEmbedder, MultimodalTextEmbedder
from haystack.document_stores.types import DocumentStore
from haystack_integrations.components.retrievers.pgvector import (
    PgvectorEmbeddingRetriever,
)

logger = logging.getLogger(__name__)


@dataclass
class PDFIndexingServiceConfig:
    document_store: DocumentStore
    vision_api_key: str
    vision_base_url: str
    vision_embeddings_model: str
    embedding_dimension: int


class IndexingService:
    """A service for indexing PDF documents using a Haystack pipeline"""

    def __init__(self, config: PDFIndexingServiceConfig):
        self.config = config
        self._init_indexing_pdf_pipeline()
        self._init_query_pipeline()

    def _init_query_pipeline(self):
        self.query_pipeline = Pipeline()

        components = {
            "embedder": MultimodalTextEmbedder(
                api_key=self.config.vision_api_key,
                api_base_url=self.config.vision_base_url,
                model=self.config.vision_embeddings_model,
            ),
            "vector_retriever": PgvectorEmbeddingRetriever(
                document_store=self.config.document_store
            ),
        }

        # Add and connect components
        for name, component in components.items():
            self.query_pipeline.add_component(name, component)

        self.query_pipeline.connect(
            "embedder.embedding", "vector_retriever.query_embedding"
        )

        logger.info("Initialized query pipeline")

    def _init_indexing_pdf_pipeline(self):
        self.pipeline = Pipeline()

        components = {
            "converter": PDFToImagesConverter(),
            "splitter": ImageSplitter(),
            "embedder": MultimodalDocumentEmbedder(
                api_key=self.config.vision_api_key,
                api_base_url=self.config.vision_base_url,
                model=self.config.vision_embeddings_model,
            ),
            "vector_writer": DocumentWriter(
                document_store=self.config.document_store,
                policy=DuplicatePolicy.OVERWRITE,
            ),
        }

        # Add and connect components
        for name, component in components.items():
            self.pipeline.add_component(name, component)

        self.pipeline.connect("converter", "splitter")
        self.pipeline.connect("splitter", "embedder")
        self.pipeline.connect("embedder", "vector_writer")

        logger.info("Initialized vision indexing pipeline")

    def index_pdf(self, file: str, metadata: Dict[str, Any] = dict()):
        try:
            output = self.pipeline.run(
                {
                    "converter": {"paths": [file], "meta": metadata},
                }
            )

            return {
                "filename": file,
                "indexed": True,
                "chunks": output.get("vector_writer", {}).get("documents_written", 0),
                "metadata": metadata,
            }
        except Exception:
            logger.error("Error processing document", exc_info=True)
            raise

    def query(self, query_text: str):
        # Sanitize query
        query_text = query_text.replace("\x00", "")
        if not query_text.strip():
            raise ValueError("Query text cannot be empty")

        try:
            output = self.query_pipeline.run(
                {
                    "embedder": {"text": query_text},
                }
            )
            documents: List[Document] = output.get("vector_retriever", {}).get(
                "documents", []
            )

            # Clean and format results
            return [
                {
                    "id": doc.id,
                    "content": (doc.content or "").replace("\x00", ""),
                    "score": float(
                        doc.score
                        if hasattr(doc, "score") and doc.score is not None
                        else 0.0
                    ),
                    "metadata": doc.meta,
                    "rank": i + 1,
                }
                for i, doc in enumerate(documents)
            ]
        except Exception as e:
            logger.error(f"Error running query pipeline: {str(e)}")
            logger.exception("Query pipeline error details:")
            raise ValueError(f"Error querying document store: {str(e)}")
