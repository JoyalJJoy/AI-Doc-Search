# DocuMind AI

AI-powered document search system (RAG-based). Upload PDFs, generate local embeddings, and ask questions to retrieve meaning-based answers.

## Overview

DocuMind AI is a full-stack project that turns scattered PDF knowledge into a searchable, semantically indexed knowledge base. It demonstrates Retrieval-Augmented Generation (RAG) fundamentals with a clean UI, Python backend, and local embedding pipeline.

## Features

- Upload PDF files
- Extract text and chunk content
- Generate embeddings locally
- Semantic search over indexed chunks
- Optional Q&A generation (OpenAI Responses API)

## Tech Stack

Frontend:
- Next.js (App Router)
- Tailwind CSS

Backend:
- FastAPI (Python)

AI / ML:
- Sentence Transformers (local embeddings)

File Handling:
- PyPDF2

## System Workflow

Upload Flow:
- User uploads a PDF
- Backend extracts text
- Text is chunked and embedded
- Chunks + embeddings stored locally

Query Flow:
- User submits a question
- Question is embedded
- System retrieves most similar chunks
- Optional answer generated with an LLM

## Project Structure

```
ai-doc-search/
  backend/
    main.py
    requirements.txt
    uploads/          # local files (git-ignored)
    storage/          # index.json (git-ignored)
  frontend/
    app/
      page.tsx
    public/
  .gitignore
```

## Setup

Backend:
```bash
cd /Users/joyaljoy/source/ai-doc-search/backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Frontend:
```bash
cd /Users/joyaljoy/source/ai-doc-search/frontend
npm install
npm run dev
```

Open your browser at `http://localhost:3000`.

## Environment Variables

Create `/Users/joyaljoy/source/ai-doc-search/backend/.env`:

```
# OpenAI option
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4.1-mini

# Gemini option
# LLM_PROVIDER=gemini
# GEMINI_API_KEY=your_gemini_api_key_here
# LLM_MODEL=gemini-2.5-flash
```

Notes:
- `/ask` supports OpenAI and Gemini.
- If `LLM_PROVIDER=gemini`, the backend uses Google's OpenAI-compatible endpoint.
- If you omit `LLM_PROVIDER`, the backend auto-detects Gemini when the model starts with `gemini` or when only `GEMINI_API_KEY` is set.
- If you want fully local mode, use `/query` only.

## API Endpoints

- `POST /upload`  
  Upload a PDF, extract text, chunk it, and store embeddings.

- `POST /query`  
  Retrieve top-K relevant chunks from indexed documents.

- `POST /ask`  
  Use retrieved chunks to generate an answer via LLM.

## Limitations

- Works best with text-based PDFs.
- Scanned PDFs are not supported yet (OCR planned).
- No vector database yet (local JSON index only).

## Contributing

We welcome contributions. Please follow these rules:

- Create a feature branch for every change.
- Keep changes focused and small.
- Update README/docs if behavior changes.
- Do not commit secrets or local files.
- Ensure code runs locally before opening a PR.
- Follow the existing code style.

## Industry Standards Checklist

To keep this repo professional and production-minded:

- Version control: clean commit history with meaningful messages.
- Branching: `feature/`, `fix/`, or `chore/` prefixes.
- Quality checks: run local tests or at least a manual smoke test.
- Security: never commit API keys or PII.
- Dependencies: keep `requirements.txt` and `package-lock.json` updated.
- Documentation: update README when features or APIs change.
- PR discipline: include a clear description and screenshots for UI changes.

## Roadmap

- OCR for scanned PDFs
- Vector DB support (FAISS or Pinecone)
- Chat-style Q&A
- Authentication and multi-user workspaces
- Cloud deployment

## License

Licensed under the Apache License, Version 2.0. See `LICENSE`.

## Author

Joyal J Joy
