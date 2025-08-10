# routes.py
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
import logging
from dotenv import load_dotenv
import os
import requests
import uuid
from typing import List, Dict
from typing import Optional

from chain_builder import get_conversational_rag_chain
from pinecone_utils import get_pinecone_retriever
from document_utils import process_and_embed_document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# --- Setup ---
router = APIRouter()
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- In-Memory Session Store ---
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Gets a chat history object for a given session_id."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# --- Build Conversational Chain ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
retriever = get_pinecone_retriever(embeddings_model)
conversational_rag_chain = get_conversational_rag_chain(retriever, get_session_history)

# --- Pydantic Models ---
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# --- API Endpoints ---

@router.post("/hackrx/run", summary="Run Submissions")
async def hackrx_run(request: HackRxRequest, Authorization: Optional[str] = Header(None)):
    """
    Processes a PDF document from a URL and answers a list of questions.
    """

    doc_url = request.documents
    questions = request.questions

    logger.info(f"Received submission request for document URL: {doc_url}")

    try:
        # Fetch the document from the URL
        response = requests.get(doc_url)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch document from URL")

        file_content = response.content
        filename = doc_url.split("/")[-1].split("?")[0]

        # Process and embed the document
        chunks_indexed = process_and_embed_document(file_content, filename)
        logger.info(f"Document '{filename}' processed and indexed successfully with {chunks_indexed} chunks.")

        # Process each question
        answers = []
        session_id = str(uuid.uuid4())
        for question in questions:
            response = conversational_rag_chain.invoke(
                {"input": question},
                config={"configurable": {"session_id": session_id}}
            )
            answers.append(response["answer"])

        return HackRxResponse(answers=answers)

    except Exception as e:
        logger.error(f"Error during submission processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")