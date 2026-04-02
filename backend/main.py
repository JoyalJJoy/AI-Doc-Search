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

# Load local embedding model (only loads once)
model = SentenceTransformer("all-MiniLM-L6-v2")

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

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "your_api_key_here":
        raise HTTPException(
            status_code=400,
            detail="OPENAI_API_KEY is missing. Add it to backend/.env or your environment.",
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="OpenAI SDK not installed. Run: pip install openai",
        ) from exc

    model_name = (payload.model or os.getenv("OPENAI_MODEL") or "gpt-4.1-mini").strip()
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
    prompt = f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"

    client = OpenAI()
    response = client.responses.create(
        model=model_name,
        instructions=instructions,
        input=prompt,
        max_output_tokens=350,
        temperature=0.2,
    )

    answer_text = getattr(response, "output_text", "") or ""

    return {
        "answer": answer_text,
        "results": results,
        "total_chunks": total_chunks,
        "model": model_name,
    }
