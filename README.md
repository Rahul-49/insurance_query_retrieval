# 📄 PDF Parser Webhook + Gemini + Pinecone 🚀

A FastAPI-based application for parsing PDF/DOCX/Text documents, embedding them using **Google Gemini**, and storing/retrieving vectors in **Pinecone**.  
Supports local API endpoint testing via **Grok** 🌐.

---

## 🛠 Features
- 📥 **Upload** PDFs/DOCX/Text via blob URL
- 🧠 **Gemini embeddings** generation
- 📦 **Pinecone vector storage**
- 💬 **Conversational RAG** for answering questions from docs
- 🔑 **Secure API key management** via `.env`

---

## 📂 Project Structure
main.py # FastAPI entrypoint
routes.py # Main API endpoints
hackrx_routes.py # HackRx-specific routes
document_utils.py # File extraction & chunking
gemini_utils.py # Gemini API helpers
pinecone_utils.py # Pinecone client helpers
requirements.txt # Dependencies


## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

2️⃣ Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

3️⃣ Install dependencies 📦
pip install -r requirements.txt

4️⃣ Create a .env file 🔑
# Gemini API Key
GEMINI_API_KEY=your_gemini_key_here

# Pinecone API Keys (from Pinecone console)
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_ENVIRONMENT=your_environment_here
PINECONE_INDEX=your_index_name_here

▶️ Running Locally
With Uvicorn

uvicorn main:app --reload
Server will start at:
📍 http://127.0.0.1:8000

🧪 Testing with Grok (Local API Endpoint)
You can test your endpoints locally using Grok or any API client (Postman, curl).
Example POST request using postman:
(Screenshot 2025-08-10 220332.png)
(Screenshot 2025-08-10 220344.png)

📜 API Endpoints
POST /api/v1/hackrx/run

## 🌐 Testing Locally with Grok
In one vscode terminal run uvicorn main:app --host 0.0.0.0 --port 8000 --reload

In other terminal you can expose your local API to the internet for quick testing using **Grok**.
**Run this command to start Grok**:
```bash
grok http 8000
```

🛡 Environment Variables & Security
Keep your .env file private (never commit to git)

API keys are loaded via python-dotenv

You can set environment variables in deployment environments as well

❤️ Credits
FastAPI for backend 🚀

Google Gemini for embeddings 🧠

Pinecone for vector storage 📦

LangChain for RAG capabilities 🔗

📌 Notes
Ensure your Pinecone index is created before running

Gemini API has rate limits — batching & retries are implemented

Document chunk size: 5000 characters for optimal semantic splitting
