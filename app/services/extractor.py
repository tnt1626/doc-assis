import os
import tempfile
import logging
import hashlib
from pathlib import Path
from docling.document_converter import DocumentConverter

logger = logging.getLogger(__name__)

_converter: DocumentConverter | None = None

def get_converter() -> DocumentConverter:
    """Lazy initialize and reuse Docling DocumentConverter instance."""
    global _converter
    if _converter is None:
        _converter = DocumentConverter()
    return _converter

def extract_document_content(content: bytes, filename: str) -> str:
    """Extract structured Markdown content from PDF or DOCX file bytes using IBM Docling.

    Args:
        content (bytes): Raw binary content of the file.
        filename (str): Name of the file (used to inspect extension .pdf or .docx).

    Returns:
        str: Structured Markdown text extracted from the document.
    """
    if not content:
        raise ValueError("The uploaded file is empty.")

    suffix = Path(filename).suffix.lower()
    if suffix not in [".pdf", ".docx"]:
        suffix = ".pdf"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
        tmp_path = tmp_file.name
        tmp_file.write(content)

    try:
        converter = get_converter()
        logger.info(
            "Extraction started: filename=%r bytes=%d sha256=%s extractor=%s",
            filename, len(content), hashlib.sha256(content).hexdigest(), __file__,
        )
        result = converter.convert(tmp_path)
        markdown_text = result.document.export_to_markdown()

        clean_text = markdown_text.strip()
        logger.info(
            "Extraction result: filename=%r status=%s input_pages=%s processed_pages=%d chars=%d errors=%s",
            filename, result.status, result.input.page_count,
            len(result.pages), len(clean_text), result.errors,
        )
        if not clean_text:
            raise ValueError("Document content is empty or text cannot be extracted.")

        return clean_text
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Docling extraction failed for '{filename}': {e}")
        raise ValueError(f"Failed to extract document '{filename}': {e}")
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

def extract_text_from_pdf(content: bytes) -> str:
    """Legacy helper function maintaining backward compatibility for PDF extraction."""
    return extract_document_content(content, "document.pdf")
