# %% [markdown]
# # Ingest the Zepto policy documents into ChromaDB
#
# This reads all 8 .txt files in docs/, splits each one into small chunks,
# turns each chunk into an embedding using a local model (no API key, no
# internet needed after the model is downloaded once), and stores those
# embeddings in a ChromaDB collection on disk so `main.py` can search them
# later.
#
# Run this once before starting the FastAPI app:
#   python ingest.py

# %%
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

THIS_FOLDER = Path(__file__).parent
DOCS_FOLDER = THIS_FOLDER / "docs"
CHROMA_PATH = THIS_FOLDER / "chroma_store"
COLLECTION_NAME = "zepto_policies"

# %% [markdown]
# ## Step 1: Chunk each document
#
# The 8 documents are short (a single paragraph each), so we treat each
# whole document as one chunk. That is a simple, valid chunking scheme
# given how short they are - a paragraph this size does not need to be
# split further to stay useful for retrieval.

# %%
def load_and_chunk_documents():
    chunks = []
    doc_files = sorted(DOCS_FOLDER.glob("doc_*.txt"))
    for doc_path in doc_files:
        text = doc_path.read_text(encoding="utf-8").strip()
        chunk_id = doc_path.stem  # e.g. "doc_01"
        chunks.append({"id": chunk_id, "text": text, "source": doc_path.name})
    return chunks


chunks = load_and_chunk_documents()
print(f"Loaded {len(chunks)} document chunks")
for c in chunks:
    print(f"  {c['id']}: {c['text'][:60]}...")

# %% [markdown]
# ## Step 2: Embed each chunk locally with sentence-transformers
#
# `all-MiniLM-L6-v2` runs entirely on your machine - the first run
# downloads the small model file, but no API key or account is needed.

# %%
print("\nLoading embedding model (all-MiniLM-L6-v2)...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chunk_texts = [c["text"] for c in chunks]
chunk_embeddings = embedding_model.encode(chunk_texts).tolist()
print(f"Created {len(chunk_embeddings)} embeddings, each of length {len(chunk_embeddings[0])}")

# %% [markdown]
# ## Step 3: Store the embeddings in a ChromaDB collection

# %%
client = chromadb.PersistentClient(path=str(CHROMA_PATH))

# start clean each time we run this, so re-running doesn't create duplicates
existing_collections = [c.name for c in client.list_collections()]
if COLLECTION_NAME in existing_collections:
    client.delete_collection(COLLECTION_NAME)

collection = client.create_collection(name=COLLECTION_NAME)

collection.add(
    ids=[c["id"] for c in chunks],
    documents=[c["text"] for c in chunks],
    embeddings=chunk_embeddings,
    metadatas=[{"source": c["source"]} for c in chunks],
)

print(f"\nStored {collection.count()} chunks in ChromaDB collection '{COLLECTION_NAME}'")
print(f"Collection saved to: {CHROMA_PATH}")

# %% [markdown]
# ## Quick sanity check - try a test search

# %%
test_query = "What is your return policy?"
test_query_embedding = embedding_model.encode([test_query]).tolist()

test_results = collection.query(query_embeddings=test_query_embedding, n_results=1)
print(f"\nTest query: '{test_query}'")
print("Top match:", test_results["documents"][0][0][:100], "...")
print("From source:", test_results["metadatas"][0][0]["source"])
