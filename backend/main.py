from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import math
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads and storage folders
BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
INDEX_PATH = os.path.join(BASE_DIR, "storage", "index.json")
ENV_PATH = os.path.join(BASE_DIR, ".env")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

def load_embedding_model() -> SentenceTransformer:
    try:
        # Prefer the local cache so backend restarts do not need network access.
        return SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)
    except OSError:
        return SentenceTransformer(EMBEDDING_MODEL_NAME)

# Load local embedding model (only loads once)
model = load_embedding_model()

# Load .env if present (simple parser to avoid extra dependencies)
def load_env_file() -> None:
    if not os.path.exists(ENV_PATH):
        return
    with open(ENV_PATH, "r", encoding="utf-8") as file:
        for line in file:
            cleaned = line.strip()
            if not cleaned or cleaned.startswith("#") or "=" not in cleaned:
                continue
            key, value = cleaned.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

load_env_file()

# Function to generate embeddings
def get_embedding(text):
    return model.encode(text).tolist()

def vector_norm(vec: List[float]) -> float:
    return math.sqrt(sum(val * val for val in vec))

def cosine_similarity(vec_a: List[float], vec_b: List[float], norm_a: float, norm_b: float) -> float:
    if norm_a == 0 or norm_b == 0:
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    return dot / (norm_a * norm_b)

def normalize_text(text: str) -> str:
    return " ".join(text.split())

def chunk_text(text: str, chunk_size: int = 220, overlap: int = 60) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks

def load_index() -> Dict[str, Any]:
    if os.path.exists(INDEX_PATH):
        try:
            with open(INDEX_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {"chunks": [], "updated_at": None}
    return {"chunks": [], "updated_at": None}

def save_index(index: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

def read_env_value(*keys: str) -> str:
    for key in keys:
        value = os.getenv(key, "").strip()
        if value and not value.startswith("your_"):
            return value
    return ""

def looks_like_gemini_key(value: str) -> bool:
    return value.startswith("AIza")

def resolve_llm_settings(requested_model: Optional[str]) -> Dict[str, Optional[str]]:
    configured_provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    configured_model = (requested_model or read_env_value("LLM_MODEL", "OPENAI_MODEL")).strip()
    openai_api_key = read_env_value("OPENAI_API_KEY")
    gemini_api_key = read_env_value("GEMINI_API_KEY")
    legacy_gemini_key = openai_api_key if looks_like_gemini_key(openai_api_key) else ""

    if configured_provider in {"openai", "gemini"}:
        provider = configured_provider
    elif configured_model.startswith("gemini"):
        provider = "gemini"
    elif gemini_api_key and not openai_api_key:
        provider = "gemini"
    elif legacy_gemini_key:
        provider = "gemini"
    else:
        provider = "openai"

    if provider == "gemini":
        api_key = gemini_api_key or legacy_gemini_key or openai_api_key
        model_name = configured_model or DEFAULT_GEMINI_MODEL
        if model_name == DEFAULT_OPENAI_MODEL:
            model_name = DEFAULT_GEMINI_MODEL
        base_url = os.getenv("GEMINI_BASE_URL", "").strip() or GEMINI_OPENAI_BASE_URL
    else:
        api_key = openai_api_key
        model_name = configured_model or DEFAULT_OPENAI_MODEL
        base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None

    return {
        "provider": provider,
        "api_key": api_key,
        "model_name": model_name,
        "base_url": base_url,
    }

def extract_chat_text(response: Any) -> str:
    choices = getattr(response, "choices", []) or []
    if not choices:
        return ""

    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""

    parts: List[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str):
            parts.append(text)
            continue
        if isinstance(text, dict):
            value = text.get("value")
            if isinstance(value, str):
                parts.append(value)
    return "\n".join(part.strip() for part in parts if part.strip())

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    filename: Optional[str] = None

class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    filename: Optional[str] = None
    model: Optional[str] = None

def retrieve_chunks(question: str, top_k: int, filename: Optional[str]) -> Tuple[List[Dict[str, Any]], int]:
    index = load_index()
    chunks = index.get("chunks", [])

    if filename:
        chunks = [chunk for chunk in chunks if chunk.get("filename") == filename]

    if not chunks:
        return [], 0

    question_embedding = get_embedding(question)
    question_norm = vector_norm(question_embedding)

    top_k = max(1, min(top_k, 10))
    scored: List[Dict[str, Any]] = []
    for chunk in chunks:
        embedding = chunk.get("embedding", [])
        embedding_norm = chunk.get("embedding_norm") or vector_norm(embedding)
        score = cosine_similarity(question_embedding, embedding, question_norm, embedding_norm)
        scored.append(
            {
                "id": chunk.get("id"),
                "text": chunk.get("text", ""),
                "score": round(score, 4),
                "filename": chunk.get("filename"),
                "page": chunk.get("page"),
                "chunk_index": chunk.get("chunk_index"),
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k], len(chunks)

@app.get("/")
def read_root():
    return {"message": "Backend is running 🚀"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    filename = os.path.basename(file.filename)
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = os.path.join(UPLOAD_FOLDER, filename)

    # Save file
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Extract text from PDF
    pages: List[Dict[str, Any]] = []
    try:
        reader = PdfReader(file_path)
        for idx, page in enumerate(reader.pages):
            pages.append(
                {
                    "page": idx + 1,
                    "text": page.extract_text() or "",
                }
            )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {e}")

    full_text = normalize_text(" ".join(page["text"] for page in pages))
    if not full_text:
        raise HTTPException(
            status_code=400,
            detail="No selectable text found. Scanned PDFs are not supported yet.",
        )

    # Build chunks per page
    doc_id = str(uuid.uuid4())
    chunks: List[Dict[str, Any]] = []
    for page in pages:
        cleaned = normalize_text(page["text"])
        for chunk_index, chunk in enumerate(chunk_text(cleaned)):
            chunks.append(
                {
                    "id": str(uuid.uuid4()),
                    "doc_id": doc_id,
                    "filename": filename,
                    "page": page["page"],
                    "chunk_index": chunk_index,
                    "text": chunk,
                }
            )

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="Unable to create chunks from this document.",
        )

    embeddings = model.encode([chunk["text"] for chunk in chunks]).tolist()
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding
        chunk["embedding_norm"] = vector_norm(embedding)

    index = load_index()
    index["chunks"] = [chunk for chunk in index["chunks"] if chunk.get("filename") != filename]
    index["chunks"].extend(chunks)
    index["updated_at"] = time.time()
    save_index(index)

    return {
        "filename": filename,
        "doc_id": doc_id,
        "pages": len(pages),
        "chunks_added": len(chunks),
        "text_preview": full_text[:300],
    }

@app.post("/query")
def query_docs(payload: QueryRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    results, total_chunks = retrieve_chunks(question, payload.top_k, payload.filename)

    if not results:
        return {"results": [], "total_chunks": 0, "message": "No documents indexed yet."}

    return {"results": results, "total_chunks": total_chunks}

@app.post("/ask")
def ask_docs(payload: AskRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    results, total_chunks = retrieve_chunks(question, payload.top_k, payload.filename)
    if not results:
        return {
            "answer": "",
            "results": [],
            "total_chunks": 0,
            "message": "No documents indexed yet.",
        }

    llm_settings = resolve_llm_settings(payload.model)
    provider = str(llm_settings["provider"])
    api_key = str(llm_settings["api_key"] or "").strip()
    model_name = str(llm_settings["model_name"] or "").strip()
    base_url = llm_settings["base_url"]

    if not api_key:
        provider_label = "Gemini" if provider == "gemini" else "OpenAI"
        env_hint = "GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY"
        raise HTTPException(
            status_code=400,
            detail=f"{provider_label} API key is missing. Add {env_hint} to backend/.env or your environment.",
        )

    try:
        from openai import APIStatusError, AuthenticationError, OpenAI, OpenAIError
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="OpenAI SDK not installed. Run: pip install openai",
        ) from exc

    context_blocks = []
    for item in results:
        source = f"{item['filename']} (page {item['page']})"
        context_blocks.append(f"[{source}]\n{item['text']}")
    context_text = "\n\n".join(context_blocks)

    instructions = (
        "You are DocuMind AI. Answer using ONLY the provided context. "
        "If the context is insufficient, say you don't know. "
        "Cite sources by filename and page in parentheses."
    )
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=350,
            temperature=0.2,
        )
    except AuthenticationError as exc:
        provider_label = "Gemini" if provider == "gemini" else "OpenAI"
        env_hint = "GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY"
        raise HTTPException(
            status_code=401,
            detail=f"{provider_label} API key is invalid or revoked. Update {env_hint} in backend/.env.",
        ) from exc
    except APIStatusError as exc:
        provider_label = "Gemini" if provider == "gemini" else "OpenAI"
        raise HTTPException(
            status_code=502,
            detail=f"{provider_label} API request failed with status {exc.status_code}.",
        ) from exc
    except OpenAIError as exc:
        provider_label = "Gemini" if provider == "gemini" else "OpenAI"
        raise HTTPException(
            status_code=502,
            detail=f"{provider_label} API request failed: {exc.__class__.__name__}.",
        ) from exc

    answer_text = extract_chat_text(response)

    return {
        "answer": answer_text,
        "results": results,
        "total_chunks": total_chunks,
        "model": model_name,
        "provider": provider,
    }
