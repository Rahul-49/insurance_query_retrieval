# hackrx_routes.py
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from typing import List
import logging
import requests
import uuid

from pinecone_utils import get_pinecone_retriever
from document_utils import process_and_embed_document
from chain_builder import get_conversational_rag_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
import os

# --- Setup ---
router = APIRouter()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Pydantic Models ---
class HackRXRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRXResponse(BaseModel):
    answers: List[str]

# --- In-Memory Session Store for webhook ---
store_webhook = {}

def get_session_history_webhook(session_id: str) -> ChatMessageHistory:
    """Gets a chat history object for a given session_id for webhook."""
    if session_id not in store_webhook:
        store_webhook[session_id] = ChatMessageHistory()
    return store_webhook[session_id]

# --- Build Conversational Chain ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
retriever = get_pinecone_retriever(embeddings_model)
conversational_rag_chain = get_conversational_rag_chain(retriever, get_session_history_webhook)

@router.post("/run", summary="Run Submissions")
async def run_submission(request: HackRXRequest, authorization: str = Header(...)):
    """
    Processes a PDF document from a URL and answers a list of questions.
    """
    expected_token = "Bearer f597b30caf9991e282500ad03f50373a9a1065fbe612947e93b5d3d4e173fa84"
    if authorization != expected_token:
        raise HTTPException(status_code=401, detail="Invalid authorization token")

    doc_url = request.documents
    questions = request.questions
    
    logger.info(f"Received submission request for document URL: {doc_url}")
    
    try:
        # Fetch the document from the URL
        response = requests.get(doc_url)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch document from URL")

        file_content = response.content
        filename = doc_url.split("/")[-1].split("?")[0] # Simple way to get filename

        # Process and embed the document
        chunks_indexed = process_and_embed_document(file_content, filename)
        logger.info(f"Document '{filename}' processed and indexed successfully with {chunks_indexed} chunks.")

        # Process each question
        answers = []
        session_id = str(uuid.uuid4()) # Use a new session for each submission
        for question in questions:
            response = conversational_rag_chain.invoke(
                {"input": question},
                config={"configurable": {"session_id": session_id}}
            )
            answers.append(response["answer"])

        return HackRXResponse(answers=answers)
    
    except Exception as e:
        logger.error(f"Error during submission processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")