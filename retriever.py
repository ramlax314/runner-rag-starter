# placeholder - will be overwritten later
# retriever.py
import os
import chromadb
from openai import OpenAI

CHROMA_PATH = "chroma"
COLLECTION_NAME = "runner_kb"
EMBED_MODEL = "text-embedding-3-small"

def embed_query(q: str):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    return client.embeddings.create(model=EMBED_MODEL, input=[q]).data[0].embedding

def retrieve_chunks(query: str, k: int = 3):
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    col = chroma_client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space":"cosine"})
    q_emb = embed_query(query)
    res = col.query(query_embeddings=[q_emb], n_results=k)
    docs = res["documents"][0] if res.get("documents") else []
    metas = res["metadatas"][0] if res.get("metadatas") else []
    return list(zip(docs, metas))
