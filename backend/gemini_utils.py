import os
import json
import time
import logging
from typing import List, Optional

import google.generativeai as genai
from dotenv import load_dotenv
import google.api_core.exceptions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Initialization ---
load_dotenv()

# Google Gemini client configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY must be set in the environment.")
genai.configure(api_key=GEMINI_API_KEY)

# --- Helper: retry wrapper for Gemini ---
def safe_generate_content(model, prompt, max_retries=3, retry_delay=2):
    for attempt in range(max_retries):
        try:
            return model.generate_content(prompt)
        except google.api_core.exceptions.ResourceExhausted:
            wait_time = retry_delay * (attempt + 1)
            logger.warning(f"[Gemini] Rate limited. Retrying in {wait_time}s...")
            time.sleep(wait_time)
        except Exception as e:
            logger.error(f"[Gemini] Error: {e}")
            raise
    raise RuntimeError("Max retries reached with Gemini API")

# --- Embeddings ---
def get_embeddings(
    texts: list,
    model="models/embedding-001",
    batch_size=10,
    delay=0.5,
    max_retries=3
) -> list:
    """
    Generates embeddings for a list of texts using the Gemini API in batches
    to reduce rate limiting. Retries on failures.
    """
    if not texts or not all(isinstance(t, str) for t in texts):
        logger.warning("get_embeddings called with invalid input.")
        return []

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for attempt in range(max_retries):
            try:
                result = genai.embed_content(
                    model=model,
                    content=batch,
                    task_type="retrieval_document"
                )

                # Gemini returns embeddings differently for single vs. multiple content
                if isinstance(result.get("embedding"), list) and isinstance(result["embedding"][0], list):
                    # Multiple embeddings
                    all_embeddings.extend(result["embedding"])
                else:
                    # Single embedding
                    all_embeddings.append(result["embedding"])

                logger.info(f"Embedded batch {i // batch_size + 1} with {len(batch)} chunks.")
                break  # success, break retry loop

            except google.api_core.exceptions.ResourceExhausted:
                wait_time = delay * (attempt + 1) * 5
                logger.warning(f"[Gemini] Rate limited on batch {i // batch_size + 1}. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            except Exception as e:
                logger.error(f"Error getting embeddings for batch {i // batch_size + 1}: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(delay * (attempt + 1))
        time.sleep(delay)  # small pause between batches

    return all_embeddings

# --- Single-question decision ---
def get_decision_from_llm(query: str, context_chunks: list) -> dict:
    """
    Generates a structured decision using Gemini, based on the query and retrieved context.
    Includes a fallback when no context is found.
    """
    if not context_chunks:
        logger.warning("No relevant context found in Pinecone. Falling back to heuristic response.")
        return {
            "Decision": "Indeterminate",
            "Amount": 0,
            "Justification": {
                "summary": "No matching clauses were found in the provided policy documents to make a decision. Please try rephrasing your question with more specific terms found in the policy.",
                "clauses": [],
                "suggested_followup": "Is the knee problem caused by an accident or was it a pre-existing condition before the policy started? Using terms like 'accident' or 'pre-existing' might yield better results."
            }
        }

    # Assemble the context from the retrieved chunks
    context = "\n---\n".join([chunk['metadata']['text'] for chunk in context_chunks])

    # Construct the prompt
    prompt = f"""
You are an expert AI assistant for processing insurance claims.
Carefully analyze the user's query and the provided policy clauses.
You MUST base your decision ONLY on the information contained within the provided policy clauses.

Relevant policy clauses:
---
{context}
---

User Query: "{query}"

Output ONLY a valid JSON object in the following format:
{{
  "Decision": "Approved" | "Rejected" | "Indeterminate",
  "Amount": <integer>,
  "Justification": {{
    "summary": "<string>",
    "clauses": [
      {{
        "clause_number": "<string>",
        "text": "<string>",
        "source_document": "<string>",
        "page_number": <integer>
      }}
    ]
  }}
}}
"""

    try:
        model = genai.GenerativeModel(
            'gemini-2.5-pro',
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=512,
                temperature=0,
                response_mime_type="application/json"
            )
        )
        response = safe_generate_content(model, prompt)
        decision_json = json.loads(response.text)
        logger.info(f"Gemini generated decision: {decision_json.get('Decision')}")
        return decision_json

    except json.JSONDecodeError:
        logger.error("Failed to parse JSON from Gemini response.")
        raise ValueError("The model returned an invalid JSON object.")
    except Exception as e:
        logger.error(f"Error getting decision from Gemini: {e}")
        raise

# --- Batch processing for multiple questions ---
def get_batch_decisions(questions: List[str], context_chunks: list) -> dict:
    """
    Processes multiple questions in a single Gemini call and returns answers in your specified format:
    {
      "answers": [
        "Answer 1",
        "Answer 2",
        ...
      ]
    }
    """
    if not questions:
        return {"answers": []}

    if not context_chunks:
        return {
            "answers": [
                "No matching policy information was found to answer this question."
                for _ in questions
            ]
        }

    context = "\n---\n".join([chunk['metadata']['text'] for chunk in context_chunks])

    prompt = f"""
You are an expert insurance policy assistant.

You will be given:
1. A set of relevant policy clauses.
2. A single or multiple questions.

Task requirements (must be followed exactly):
- Use ONLY the provided policy clauses to answer each question. Do NOT use outside knowledge.
- Answer each question in **one to two clear, grammatically correct sentences** in a formal insurance-policy tone.
- Preserve numeric and legal notation exactly (e.g., "thirty-six (36) months", "two (2) years", "5%").
- Do NOT include any extra commentary, meta-text, or explanations.
- Do NOT produce bullet points, lists, headings, or markdown; produce plain sentences.
- Avoid using quotation marks or slashes around clause text; paraphrase clause fragments rather than quoting them.
- No need to include page numbers in the answers.
- If the policy does not contain the answer, respond with exactly: "Information not available in the provided policy clauses."
- The model output MUST be a single valid JSON object **and nothing else** in the exact format below:
Policy clauses:
---
{context}
---

Questions:
{json.dumps(questions)}

Return only a valid JSON object in the exact format below:
{{
  "answers":
    "<Answer to Question 1 in formal plain text style>",
}}
"""

    model = genai.GenerativeModel(
        'gemini-2.5-pro',
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=256,
            temperature=0,
            response_mime_type="application/json"
        )
    )
    response = safe_generate_content(model, prompt)
    return json.loads(response.text)
