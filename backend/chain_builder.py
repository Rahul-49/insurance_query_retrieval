# chain_builder.py
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

# --- Load environment variables to get the API key ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

def get_conversational_rag_chain(retriever, get_session_history):
    """
    Builds and returns a conversational RAG chain with session history.
    """
    # --- FIXED: Changed the system prompt to ask for a direct Yes/No style answer ---
    qa_system_prompt = """You are an insurance claim adjudication bot. Your task is to give a direct and concise answer.
Analyze the user's query and the provided context.
Based ONLY on the context, answer with "Yes, [the requested procedure/claim] is covered under the policy." or "No, [the requested procedure/claim] is not covered under the policy."
If you have enough information for a "Yes" or "No" answer, follow it with a explanation for your decision with supporting points from pdf as page numbers and line.
Do not add any extra conversational text, greetings, or apologies.

--- CONTEXT ---
{context}
-----------------"""

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    # Chain that combines documents into a single string and passes to the LLM
    Youtube_chain = create_stuff_documents_chain(
        llm=ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GEMINI_API_KEY,
            convert_system_message_to_human=True
        ),
        prompt=qa_prompt
    )

    # 2. System prompt for reformulating the question based on history
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question which can be understood \
without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    # Retriever that is aware of the chat history
    history_aware_retriever = create_history_aware_retriever(
        llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY),
        retriever=retriever,
        prompt=contextualize_q_prompt
    )

    # 3. The final retrieval chain
    rag_chain = create_retrieval_chain(history_aware_retriever, Youtube_chain)

    # 4. Wrap the chain with session history management
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag_chain