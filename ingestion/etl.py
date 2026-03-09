# ingestion/etl.py
import os
import re
import sys
import html as ihtml
import argparse
from datetime import datetime, timezone
from typing import List, Dict, Iterable, Tuple

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import weaviate
from weaviate.util import generate_uuid5

# Reuse your Confluence client (VPN proxy-aware)
# Make sure your project has: confluence/client.py with ConfluenceClient class
from confluence.client import ConfluenceClient

load_dotenv()


# -----------------------------
# Settings / Defaults
# -----------------------------
DEF_WEAVIATE_ENDPOINT = os.getenv("WEAVIATE_ENDPOINT", "http://localhost:8081")
DEF_WEAVIATE_CLASS = os.getenv("WEAVIATE_CLASS", "ConfluencePageChunk")
DEF_EMB_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

DEF_ROOT_SPACE = os.getenv("ROOT_PAGE_SPACE", "EE")
DEF_ROOT_TITLE = os.getenv("ROOT_PAGE_TITLE", "End-to-End Test Onbox Team Germany")

DEF_CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "450"))
DEF_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP_WORDS", "60"))


# -----------------------------
# Weaviate schema ensure
# -----------------------------
def ensure_weaviate_schema(client: weaviate.Client, class_name: str):
    schema = client.schema.get()
    classes = [c["class"] for c in schema.get("classes", [])]
    if class_name in classes:
        return

    client.schema.create_class({
        "class": class_name,
        "description": "Chunked Confluence content with metadata for semantic search",
        "vectorizer": "none",
        "vectorIndexType": "hnsw",
        "vectorIndexConfig": {"distance": "cosine"},
        "invertedIndexConfig": {"stopwords": {"preset": "en"}},
        "properties": [
            {"name": "page_id",           "dataType": ["text"],   "indexInverted": True},
            {"name": "chunk_id",          "dataType": ["text"],   "indexInverted": True},
            {"name": "title",             "dataType": ["text"],   "indexInverted": True},
            {"name": "snippet",           "dataType": ["text"],   "indexInverted": True},
            {"name": "body_text",         "dataType": ["text"],   "indexInverted": True},
            {"name": "last_modified",     "dataType": ["date"],   "indexInverted": True},
            {"name": "version_number",    "dataType": ["int"],    "indexInverted": True},
            {"name": "author",            "dataType": ["text"],   "indexInverted": True},
            {"name": "space_key",         "dataType": ["text"],   "indexInverted": True},
            {"name": "space_name",        "dataType": ["text"],   "indexInverted": True},
            {"name": "url",               "dataType": ["text"],   "indexInverted": False},
            {"name": "highlighted_title", "dataType": ["text"],   "indexInverted": False},
            {"name": "timestamp",         "dataType": ["number"], "indexInverted": True},
        ],
    })
    print(f"[schema] Created class '{class_name}'")


# -----------------------------
# HTML → plain text cleaner
# -----------------------------
TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")

def clean_html_to_text(html: str) -> str:
    """
    Basic cleaner that removes tags, normalizes whitespace, and unescapes entities.
    You can later replace with BeautifulSoup if needed.
    """
    if not html:
        return ""
    text = TAG_RE.sub(" ", html)
    text = WS_RE.sub(" ", text).strip()
    return ihtml.unescape(text)


# -----------------------------
# Chunking
# -----------------------------
def chunk_by_words(text: str, max_words: int = 450, overlap: int = 60) -> List[str]:
    """
    Split text into overlapping word chunks. Simple, fast, and good enough for MVP.
    """
    if not text:
        return []
    words = text.split()
    if len(words) <= max_words:
        return [" ".join(words)]

    chunks = []
    start = 0
    step = max_words - overlap
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step
    return chunks


def make_snippet(text: str, limit: int = 300) -> str:
    return (text[:limit] + "…") if text and len(text) > limit else (text or "")


# -----------------------------
# Confluence helpers
# -----------------------------
def extract_page_metadata(page_json: dict) -> Dict:
    """
    Pull the fields we care about from the full page JSON:
    title, id, space (key/name), version.when/number/by, body.storage.value, url, etc.
    """
    content = page_json or {}

    title = content.get("title") or ""
    page_id = content.get("id") or ""

    space = (content.get("space") or {})
    space_key = space.get("key") or ""
    space_name = space.get("name") or ""

    version = (content.get("version") or {})
    last_modified = version.get("when")
    version_number = version.get("number") or 0
    author = (version.get("by") or {}).get("displayName") or ""

    links = content.get("_links") or {}
    url = links.get("webui") or ""  # relative (Confluence adds base at runtime)

    body = (content.get("body") or {}).get("storage") or {}
    html_value = body.get("value") or ""
    body_text = clean_html_to_text(html_value)

    # Derive timestamp (epoch ms) if we have ISO date
    timestamp = None
    if last_modified:
        try:
            dt = datetime.fromisoformat(last_modified.replace("Z", "+00:00"))
            timestamp = int(dt.timestamp() * 1000)
        except Exception:
            timestamp = None

    return {
        "page_id": page_id,
        "title": title,
        "space_key": space_key,
        "space_name": space_name,
        "last_modified": last_modified,
        "version_number": version_number,
        "author": author,
        "url": url,
        "body_text": body_text,
        "timestamp": timestamp,
    }


# -----------------------------
# Ingestion orchestration
# -----------------------------
def upsert_chunks(
    wclient: weaviate.Client,
    class_name: str,
    page_meta: Dict,
    chunks: List[str],
    model: SentenceTransformer
) -> int:
    """
    Embed each chunk and upsert to Weaviate using the batch API.
    Returns number of chunks inserted.
    """
    if not chunks:
        return 0

    # Precompute vectors
    vectors = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    if isinstance(vectors, list):
        vectors = np.array(vectors, dtype="float32")
    else:
        vectors = vectors.astype("float32")

    # Prepare batched upserts
    count = 0
    with wclient.batch(batch_size=64) as batch:
        for idx, chunk_text in enumerate(chunks):
            obj = {
                "page_id": page_meta["page_id"],
                "chunk_id": f'{page_meta["page_id"]}_{idx:04d}',
                "title": page_meta["title"],
                "snippet": make_snippet(chunk_text, 280),
                "body_text": chunk_text,
                "last_modified": page_meta["last_modified"],
                "version_number": page_meta["version_number"],
                "author": page_meta["author"],
                "space_key": page_meta["space_key"],
                "space_name": page_meta["space_name"],
                "url": page_meta["url"],
                "highlighted_title": page_meta["title"],  # keep as-is; you can add highlights later
                "timestamp": page_meta["timestamp"] if page_meta["timestamp"] is not None else 0,
            }

            obj_id = generate_uuid5(obj["page_id"] + obj["chunk_id"])
            batch.add_data_object(
                data_object=obj,
                class_name=class_name,
                uuid=obj_id,
                vector=vectors[idx].tolist()
            )
            count += 1

    return count


async def run_etl(
    root_space: str,
    root_title: str,
    weaviate_endpoint: str,
    class_name: str,
    model_name: str,
    chunk_words: int,
    chunk_overlap: int,
    dry_run: bool = False,
    limit_pages: int = None,
):
    """
    Main ETL:
      1) Resolve root page ID
      2) Get descendants (page ids)
      3) For each page: fetch, extract metadata, clean, chunk, embed, upsert
    """
    # Weaviate client
    wclient = weaviate.Client(weaviate_endpoint)
    ensure_weaviate_schema(wclient, class_name)

    # Embedding model
    print(f"[embeddings] Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    # Confluence client (uses your VPN auto-proxy logic)
    cf = ConfluenceClient()

    # 1) Resolve root
    print(f"[root] Resolving page ID for: space='{root_space}', title='{root_title}'")
    root_res = await cf.get_page_id(root_space, root_title)
    if not root_res:
        print("[root] Not found. Check space key and exact title.")
        return

    root_id = root_res["id"]
    print(f"[root] page_id = {root_id}")

    # 2) Descendants
    print("[desc] Fetching descendants…")
    descendants = await cf.get_descendants(root_id, max_nodes=5000)
    print(f"[desc] Found {len(descendants)} nodes under root")

    if limit_pages:
        descendants = descendants[:limit_pages]
        print(f"[desc] Limiting to first {limit_pages} pages for this run")

    total_chunks = 0
    total_pages = 0

    # 3) For each page → fetch full content
    for n, node in enumerate(descendants, start=1):
        pid = node.get("id")
        try:
            page_json = await cf.get_page(pid)
        except Exception as ex:
            print(f"[page] ERROR fetching id={pid}: {ex}")
            continue

        meta = extract_page_metadata(page_json)
        if not meta["body_text"]:
            print(f"[page] Skipping empty body: id={pid} title='{meta['title']}'")
            continue

        if dry_run:
            # Just show what we'd do
            print(f"[dry-run] Would index page id={pid} title='{meta['title']}' words={len(meta['body_text'].split())}")
            total_pages += 1
            continue

        # chunk
        chunks = chunk_by_words(meta["body_text"], max_words=chunk_words, overlap=chunk_overlap)
        # embed + upsert to Weaviate
        try:
            inserted = upsert_chunks(
                wclient=wclient,
                class_name=class_name,
                page_meta=meta,
                chunks=chunks,
                model=model
            )
            total_chunks += inserted
            total_pages += 1
            print(f"[ok] {n}/{len(descendants)} id={pid} chunks={inserted} title='{meta['title']}'")
        except Exception as ex:
            print(f"[weaviate] ERROR upserting id={pid}: {ex}")

    print("------------------------------------------------")
    print(f"[summary] pages indexed: {total_pages}")
    print(f"[summary] chunks indexed: {total_chunks}")
    print("[done]")


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Confluence → Weaviate ETL")
    p.add_argument("--space", default=DEF_ROOT_SPACE, help="Root space key (e.g., EE)")
    p.add_argument("--title", default=DEF_ROOT_TITLE, help="Root page title (exact)")
    p.add_argument("--weaviate-endpoint", default=DEF_WEAVIATE_ENDPOINT, help="Weaviate HTTP endpoint")
    p.add_argument("--class-name", default=DEF_WEAVIATE_CLASS, help="Weaviate class name")
    p.add_argument("--model", default=DEF_EMB_MODEL, help="Embedding model (sentence-transformers)")
    p.add_argument("--chunk-words", type=int, default=DEF_CHUNK_WORDS, help="Max words per chunk")
    p.add_argument("--chunk-overlap", type=int, default=DEF_CHUNK_OVERLAP, help="Overlap words between chunks")
    p.add_argument("--dry-run", action="store_true", help="Do not write to DB; just log actions")
    p.add_argument("--limit-pages", type=int, default=None, help="Optional limit for pages processed in this run")
    return p.parse_args()


if __name__ == "__main__":
    # Run async entrypoint
    import asyncio
    args = parse_args()
    try:
        asyncio.run(
            run_etl(
                root_space=args.space,
                root_title=args.title,
                weaviate_endpoint=args.weaviate_endpoint,
                class_name=args.class_name,
                model_name=args.model,
                chunk_words=args.chunk_words,
                chunk_overlap=args.chunk_overlap,
                dry_run=args.dry_run,
                limit_pages=args.limit_pages,
            )
        )
    except KeyboardInterrupt:
        print("\n[abort] Interrupted by user")
        sys.exit(1)
