"use client";
import { useState } from "react";

type UploadInfo = {
  filename: string;
  doc_id: string;
  pages: number;
  chunks_added: number;
  text_preview: string;
};

type SearchResult = {
  id: string;
  text: string;
  score: number;
  filename: string;
  page: number;
  chunk_index: number;
};

type SearchPayload = {
  question: string;
  top_k: number;
  filename?: string;
};

type QueryResponse = {
  results: SearchResult[];
  total_chunks: number;
  message?: string;
};

type AskResponse = QueryResponse & {
  answer: string;
  model?: string;
  provider?: string;
};

const API_BASE = "http://127.0.0.1:8000";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [uploadInfo, setUploadInfo] = useState<UploadInfo | null>(null);
  const [uploadError, setUploadError] = useState("");
  const [isUploading, setIsUploading] = useState(false);

  const [question, setQuestion] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [searchError, setSearchError] = useState("");
  const [activeAction, setActiveAction] = useState<"query" | "ask" | null>(null);
  const [scopeToLatest, setScopeToLatest] = useState(true);
  const [answer, setAnswer] = useState("");
  const [modelUsed, setModelUsed] = useState("");

  const handleUpload = async () => {
    setUploadError("");
    setSearchError("");
    setResults([]);
    setAnswer("");
    setModelUsed("");

    if (!file) {
      setUploadError("Choose a PDF file before uploading.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setIsUploading(true);
    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (!res.ok) {
        setUploadError(data.detail || "Upload failed.");
        return;
      }
      if (data.error) {
        setUploadError(data.error);
        return;
      }

      setUploadInfo(data);
      setAnswer("");
      setModelUsed("");
    } catch (err) {
      console.error(err);
      setUploadError("Upload failed.");
    } finally {
      setIsUploading(false);
    }
  };

  const buildSearchPayload = (): SearchPayload | null => {
    setSearchError("");
    if (!question.trim()) {
      setSearchError("Type a question to search.");
      return null;
    }

    const payload: SearchPayload = {
      question: question.trim(),
      top_k: 5,
    };

    if (scopeToLatest && uploadInfo?.filename) {
      payload.filename = uploadInfo.filename;
    }

    return payload;
  };

  const handleSemanticSearch = async () => {
    const payload = buildSearchPayload();
    setAnswer("");
    setModelUsed("");

    if (!payload) {
      return;
    }

    setActiveAction("query");
    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = (await res.json()) as QueryResponse & { detail?: string; error?: string };
      if (!res.ok) {
        setSearchError(data.detail || "Semantic search failed.");
        return;
      }
      if (data.error) {
        setSearchError(data.error);
        return;
      }

      setResults(data.results || []);
    } catch (err) {
      console.error(err);
      setSearchError("Semantic search failed.");
    } finally {
      setActiveAction(null);
    }
  };

  const handleGenerateAnswer = async () => {
    const payload = buildSearchPayload();
    setSearchError("");
    setAnswer("");
    setModelUsed("");

    if (!payload) {
      return;
    }

    setActiveAction("ask");
    try {
      const res = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = (await res.json()) as AskResponse & { detail?: string; error?: string };
      if (!res.ok) {
        setSearchError(data.detail || "Answer generation failed.");
        return;
      }
      if (data.error) {
        setSearchError(data.error);
        return;
      }

      setResults(data.results || []);
      setAnswer(data.answer || "");
      setModelUsed(data.model || "");
    } catch (err) {
      console.error(err);
      setSearchError("Answer generation failed.");
    } finally {
      setActiveAction(null);
    }
  };

  return (
    <div className="min-h-screen app-shell">
      <div className="mx-auto flex w-full max-w-5xl flex-col gap-10 px-6 py-12">
        <header className="flex flex-col gap-4">
          <span className="text-xs uppercase tracking-[0.38em] text-emerald-800">
            DocuMind AI
          </span>
          <h1 className="text-4xl font-[var(--font-display)] leading-tight text-slate-900 md:text-6xl">
            Turn PDFs into searchable knowledge.
          </h1>
          <p className="max-w-2xl text-base text-slate-700 md:text-lg">
            Upload a document, generate embeddings locally, and explore meaning-first
            answers in seconds.
          </p>
        </header>

        <div className="grid gap-8 lg:grid-cols-[1.1fr_0.9fr]">
          <section className="panel flex flex-col gap-6 rounded-3xl p-6 md:p-8">
            <div className="flex items-center justify-between">
              <h2 className="text-2xl font-[var(--font-display)] text-slate-900">
                1. Upload & Index
              </h2>
              <span className="rounded-full border border-emerald-700/30 bg-emerald-50 px-3 py-1 text-xs text-emerald-800">
                PDF only
              </span>
            </div>

            <div className="flex flex-col gap-3">
              <input
                type="file"
                accept="application/pdf"
                className="w-full rounded-2xl border border-slate-300/70 bg-white/70 px-4 py-3 text-sm file:mr-3 file:rounded-full file:border-0 file:bg-emerald-700 file:px-4 file:py-2 file:text-sm file:text-white hover:file:bg-emerald-800"
                onChange={(e) => {
                  const selectedFile = e.target.files?.[0] || null;
                  setFile(selectedFile);
                  if (selectedFile) {
                    setUploadInfo(null);
                    setResults([]);
                    setAnswer("");
                    setModelUsed("");
                  }
                }}
              />
              <div className="text-sm text-slate-700">
                {file ? (
                  <span className="text-emerald-800">
                    Selected: <strong>{file.name}</strong>
                  </span>
                ) : (
                  <span className="text-amber-700">No file selected yet.</span>
                )}
              </div>
            </div>

            <button
              onClick={handleUpload}
              disabled={isUploading}
              className="w-full rounded-2xl bg-emerald-700 px-6 py-3 text-sm font-semibold text-white transition hover:bg-emerald-800 disabled:cursor-not-allowed disabled:bg-emerald-400"
            >
              {isUploading ? "Indexing..." : "Upload & Build Embeddings"}
            </button>

            {uploadError && (
              <div className="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                {uploadError}
              </div>
            )}

            {uploadInfo && (
              <div className="flex flex-col gap-4 rounded-2xl border border-slate-200/80 bg-white/70 p-4">
                <div className="flex flex-wrap gap-4 text-xs uppercase tracking-[0.25em] text-slate-500">
                  <span>{uploadInfo.filename}</span>
                  <span>{uploadInfo.pages} pages</span>
                  <span>{uploadInfo.chunks_added} chunks</span>
                </div>
                <p className="text-sm text-slate-700">Preview</p>
                <p className="text-sm leading-relaxed text-slate-800">
                  {uploadInfo.text_preview}
                </p>
              </div>
            )}
          </section>

          <section className="panel flex flex-col gap-6 rounded-3xl p-6 md:p-8">
            <h2 className="text-2xl font-[var(--font-display)] text-slate-900">
              2. Search Or Ask
            </h2>
            <div className="flex flex-col gap-3">
              <input
                type="text"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="What does the contract say about termination?"
                className="w-full rounded-2xl border border-slate-300/70 bg-white/70 px-4 py-3 text-sm text-slate-800"
              />
              <label className="flex items-center gap-2 text-xs text-slate-600">
                <input
                  type="checkbox"
                  checked={scopeToLatest}
                  onChange={(e) => setScopeToLatest(e.target.checked)}
                  className="h-4 w-4 accent-emerald-700"
                />
                Search only the last uploaded document
              </label>
            </div>

            <div className="grid gap-3 md:grid-cols-2">
              <button
                onClick={handleSemanticSearch}
                disabled={activeAction !== null}
                className="w-full rounded-2xl border border-slate-300/80 bg-white px-6 py-3 text-sm font-semibold text-slate-900 transition hover:border-slate-400 hover:bg-slate-50 disabled:cursor-not-allowed disabled:border-slate-200 disabled:bg-slate-100 disabled:text-slate-400"
              >
                {activeAction === "query" ? "Searching..." : "Run Semantic Search"}
              </button>
              <button
                onClick={handleGenerateAnswer}
                disabled={activeAction !== null}
                className="w-full rounded-2xl bg-slate-900 px-6 py-3 text-sm font-semibold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-500"
              >
                {activeAction === "ask" ? "Generating..." : "Generate Answer"}
              </button>
            </div>

            {searchError && (
              <div className="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                {searchError}
              </div>
            )}

            {answer && (
              <div className="rounded-2xl border border-emerald-200 bg-emerald-50/70 px-4 py-4 text-sm text-emerald-900">
                <div className="text-xs uppercase tracking-[0.25em] text-emerald-700">
                  Answer {modelUsed ? `· ${modelUsed}` : ""}
                </div>
                <p className="mt-3 whitespace-pre-line leading-relaxed">{answer}</p>
              </div>
            )}

            <div className="text-xs text-slate-600">
              Semantic search is fully local. Generate Answer uses your configured AI
              provider only after the top matches are retrieved.
            </div>
          </section>
        </div>

        <section className="panel flex flex-col gap-4 rounded-3xl p-6 md:p-8">
          <div className="flex items-center justify-between">
            <h3 className="text-xl font-[var(--font-display)] text-slate-900">
              Top Matches
            </h3>
            <span className="text-xs uppercase tracking-[0.3em] text-slate-500">
              {results.length} results
            </span>
          </div>

          {results.length === 0 ? (
            <div className="rounded-2xl border border-dashed border-slate-300/80 bg-white/60 px-4 py-6 text-sm text-slate-600">
              Upload a PDF and run a query to see matching chunks here.
            </div>
          ) : (
            <div className="grid gap-4">
              {results.map((result) => (
                <div
                  key={result.id}
                  className="rounded-2xl border border-slate-200/80 bg-white/80 px-5 py-4"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-500">
                    <span>{result.filename}</span>
                    <span>Page {result.page}</span>
                    <span>Score {Number(result.score).toFixed(3)}</span>
                  </div>
                  <p className="mt-3 text-sm leading-relaxed text-slate-800">
                    {result.text.length > 280
                      ? `${result.text.slice(0, 280)}...`
                      : result.text}
                  </p>
                </div>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
