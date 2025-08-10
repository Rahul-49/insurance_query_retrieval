# document_utils.py
import os
import tempfile
import pdfplumber
import docx
import logging
from nltk.tokenize import sent_tokenize
from typing import List

# Import the specific functions you need from the new gemini_utils.py
from gemini_utils import get_embeddings
from pinecone_utils import upsert_vectors

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _extract_text(file_bytes, filename: str) -> str:
    """Helper function to extract text from different file types."""
    suffix = os.path.splitext(filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    text = ""
    try:
        if suffix == ".pdf":
            with pdfplumber.open(tmp_path) as pdf:
                text = "\n".join([page.extract_text(x_tolerance=1) or "" for page in pdf.pages])
                logger.info(f"Extracted {len(text)} characters from PDF: {filename}")
        elif suffix == ".docx":
            doc = docx.Document(tmp_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            logger.info(f"Extracted {len(text)} characters from DOCX: {filename}")
        else:
            text = file_bytes.decode("utf-8")
            logger.info(f"Read {len(text)} characters from text file: {filename}")
    finally:
        os.remove(tmp_path)

    return text

def _chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """
    Splits text into chunks using sentence boundaries for semantic integrity.
    """
    if not text:
        return []

    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def process_and_embed_document(file_bytes, filename: str) -> int:
    """
    The main processing pipeline for a single document.
    Extracts text, chunks it, creates embeddings with Gemini, and upserts to Pinecone.
    """
    logger.info(f"Starting processing for document: {filename}")

    text = _extract_text(file_bytes, filename)
    if not text.strip():
        logger.warning(f"No text extracted from {filename}. Skipping.")
        return 0

    chunks = _chunk_text(text)
    logger.info(f"Split text into {len(chunks)} chunks.")

    if not chunks:
        return 0

    try:
        embeddings = get_embeddings(chunks)
        if not embeddings:
            logger.warning(f"No embeddings were generated for {filename}.")
            return 0
    except Exception as e:
        logger.error(f"Failed to create embeddings for {filename}: {e}")
        return 0

    vectors_to_upsert = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_id = f"{filename}-{i}"
        metadata = {"text": chunk, "source": filename, "page_number": 0}
        vectors_to_upsert.append((chunk_id, embedding, metadata))

    if vectors_to_upsert:
        upsert_vectors(vectors_to_upsert)

    return len(vectors_to_upsert)