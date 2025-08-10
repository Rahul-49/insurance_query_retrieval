# pinecone_utils.py
import os
from pinecone import Pinecone
from dotenv import load_dotenv
import logging
from langchain_pinecone import PineconeVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Initialization ---
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX")

try:
    if not all([PINECONE_API_KEY, PINECONE_INDEX_NAME]):
        raise ValueError("PINECONE_API_KEY and PINECONE_INDEX must be set in the environment.")
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    index_stats = index.describe_index_stats()
    logger.info(f"Successfully connected to Pinecone index '{PINECONE_INDEX_NAME}'. Stats: {index_stats}")
    
except Exception as e:
    logger.error(f"Failed to initialize Pinecone: {e}")
    index = None

def get_pinecone_retriever(embeddings):
    """
    Returns a LangChain retriever for the Pinecone index.
    """
    if not index:
        raise ConnectionError("Pinecone index is not initialized.")
    
    vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
    return vectorstore.as_retriever()


def upsert_vectors(vectors: list):
    if not index:
        raise ConnectionError("Pinecone index is not initialized.")
    if not vectors:
        logger.warning("Upsert called with no vectors.")
        return
        
    try:
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)
            logger.info(f"Successfully upserted batch {i//batch_size + 1}")
        logger.info("Upsert process completed.")
    except Exception as e:
        logger.error(f"Error during Pinecone upsert: {e}")
        raise