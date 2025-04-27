import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

import pymupdf
from haystack import Document, component

logger = logging.getLogger(__name__)


@component
class PDFToImagesConverter:
    """Converts PDFs to a Haystack Document comprising of a list of images"""

    def __init__(
        self,
        output_dir: str = "./pdf_images",
        dpi: int = 75,
        format: str = "jpg",
        prefix: str = "page_",
    ):
        """
        Initialize the PDF to Images converter

        Args:
            output_dir: Directory to save the generated images
            dpi: Resolution of the output images
            format: Image format (png, jpg, etc.)
            prefix: Prefix for image filenames
        """
        self.output_dir = output_dir
        self.dpi = dpi
        self.format = format
        self.prefix = prefix

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    @component.output_types(image_paths=List[str], documents=List[Document])
    def run(
        self, paths: List[str], meta: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Convert PDF files to images of each page

        Args:
            paths: List of PDF file paths to process
            meta: Optional metadata to attach to documents

        Returns:
            Dict containing list of image paths and document objects
        """
        if meta is None:
            meta = {}

        documents = []

        for pdf_path in paths:
            try:
                if not pdf_path.lower().endswith(".pdf"):
                    logger.warning(f"Skipping non-PDF file: {pdf_path}")
                    continue

                # Create a subdirectory for this PDF's images
                pdf_name = Path(pdf_path).stem
                # Add a unique identifier to the output directory in case of multiple PDFs with the
                # same name
                pdf_name = f"{pdf_name}_{uuid.uuid4()}"
                pdf_output_dir = os.path.join(self.output_dir, pdf_name)
                os.makedirs(pdf_output_dir, exist_ok=True)

                # Open the PDF
                pdf_document = pymupdf.open(pdf_path)
                pdf_image_paths = []

                # Convert each page to an image
                for page_num, page in enumerate(pdf_document):
                    # Render page to a pixmap
                    pix = page.get_pixmap(dpi=self.dpi)

                    # Save the image
                    image_path = os.path.join(
                        pdf_output_dir, f"{self.prefix}{page_num + 1}.{self.format}"
                    )
                    pix.save(image_path)
                    pdf_image_paths.append(image_path)

                # Create a document with metadata
                doc_meta = meta.copy()
                doc_meta["file_path"] = pdf_path
                doc_meta["image_paths"] = pdf_image_paths
                doc_meta["page_count"] = len(pdf_image_paths)

                doc_content = (
                    f"PDF converted to {len(pdf_image_paths)} images: {pdf_path}"
                )
                documents.append(Document(content=doc_content, meta=doc_meta))

                logger.info(
                    f"Successfully converted {pdf_path} to {len(pdf_image_paths)} images"
                )

            except Exception as e:
                logger.error(f"Failed to process PDF file {pdf_path}: {str(e)}")
                raise RuntimeError(f"Failed to process PDF file {pdf_path}: {str(e)}")

        return {"documents": documents}
