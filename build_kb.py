# placeholder - will be overwritten later
# build_kb.py
import os, re, uuid
from typing import List, Tuple
import chromadb
from chromadb.config import Settings
from openai import OpenAI

KB_DIR = "knowledge"
CHROMA_PATH = "chroma"
COLLECTION_NAME = "runner_kb"
EMBED_MODEL = "text-embedding-3-small"

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_docx(path: str) -> str:
    try:
        from docx import Document
    except Exception:
        return ""
    try:
        doc = Document(path)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception:
        return ""

def collect_texts(kb_dir: str) -> List[Tuple[str, str]]:
    out = []
    for name in sorted(os.listdir(kb_dir)):
        path = os.path.join(kb_dir, name)
        if not os.path.isfile(path): 
            continue
        if name.lower().endswith((".md", ".txt")):
            out.append((name, read_text(path)))
        elif name.lower().endswith(".docx"):
            txt = read_docx(path)
            if txt.strip():
                out.append((name, txt))
    return out

def simple_chunk(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    text = re.sub(r"\r\n?", "\n", text).strip()
    parts = re.split(r"\n\s*\n", text)
    parts = [p.strip() for p in parts if p.strip()]

    chunks, buf = [], ""
    for part in parts:
        if len(buf) + len(part) + 1 <= max_chars:
            buf += ("\n" + part) if buf else part
        else:
            if buf: chunks.append(buf)
            for i in range(0, len(part), max_chars - overlap):
                piece = part[i:i + (max_chars - overlap)]
                chunks.append(piece)
            buf = ""
    if buf:
        chunks.append(buf)

    overlapped = []
    for i, ch in enumerate(chunks):
        if i == 0:
            overlapped.append(ch)
        else:
            prev_tail = chunks[i-1][-overlap:]
            overlapped.append(prev_tail + "\n" + ch)
    return overlapped

def rebuild_kb() -> int:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")

    client = OpenAI(api_key=api_key)
    sources = collect_texts(KB_DIR)

    documents, metadatas = [], []
    for fname, raw in sources:
        if not raw.strip():
            continue
        chunks = simple_chunk(raw, max_chars=1200, overlap=200)
        for idx, ch in enumerate(chunks):
            documents.append(ch)
            metadatas.append({"source": fname, "chunk": idx})

    if not documents:
        return 0

    resp = client.embeddings.create(model=EMBED_MODEL, input=documents)
    vectors = [d.embedding for d in resp.data]

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    col = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space":"cosine"}
    )
    if col.count() > 0:
        col.delete(where={})

    ids = [str(uuid.uuid4()) for _ in documents]
    col.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=vectors)
    return len(documents)

def collection_count() -> int:
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        col = chroma_client.get_collection(COLLECTION_NAME)
        return col.count()
    except Exception:
        return 0

if __name__ == "__main__":
    n = rebuild_kb()
    print(f"Built KB with {n} chunks.")
