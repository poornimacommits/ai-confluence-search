# display_results.py
import os
import re
import json
import unicodedata
import numpy as np
from datetime import datetime, timezone
import weaviate
from dotenv import load_dotenv
from openai import AzureOpenAI


# -------------------------------------------------------
# 1) Load environment
# -------------------------------------------------------
load_dotenv()

AOAI_API_KEY   = os.getenv("AZURE_OPENAI_KEY")
AOAI_ENDPOINT  = os.getenv("AZURE_OPENAI_ENDPOINT")
AOAI_VERSION   = os.getenv("AZURE_OPENAI_API_VERSION")
AOAI_MODEL     = os.getenv("AZURE_OPENAI_QUERY_REWRITE_DEPLOYMENT")  # e.g., "gpt-4o"

if not AOAI_API_KEY or not AOAI_ENDPOINT or not AOAI_VERSION or not AOAI_MODEL:
    raise RuntimeError("Missing Azure OpenAI ENV vars: AZURE_OPENAI_KEY / AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_VERSION / AZURE_OPENAI_QUERY_REWRITE_DEPLOYMENT")


# -------------------------------------------------------
# 2) Azure OpenAI client (GPT‑4o)
# -------------------------------------------------------
client_llm = AzureOpenAI(
    azure_endpoint=AOAI_ENDPOINT,
    api_key=AOAI_API_KEY,
    api_version=AOAI_VERSION,
)


# -------------------------------------------------------
# 3) Weaviate v3 client
# -------------------------------------------------------
client_wv = weaviate.Client(
    url="http://localhost:8080",
    additional_headers={
        # Only needed if your text2vec-openai module requires headers at query time
        "X-Openai-Api-Key": AOAI_API_KEY,
        "X-Azure-Api-Key": AOAI_API_KEY,
    }
)


# -------------------------------------------------------
# 4) Random sampling from Weaviate (for testing)
# -------------------------------------------------------
def get_random_documents(limit=20):
    """
    Retrieve ~random documents by using a random vector search.
    Works with weaviate-client 3.26.6.
    """
    random_vec = np.random.rand(1536).tolist()

    response = (
        client_wv.query
        .get(
            "Metadata",
            [
                "title",
                "snippet",
                "url",
                "timestamp",
                "created_timestamp",
                "author",
                "created_by",
                "space_name",
                "ancestor_titles",
                "version_number",
            ]
        )
        .with_near_vector({"vector": random_vec})
        .with_limit(limit)
        .do()
    )

    return response.get("data", {}).get("Get", {}).get("Metadata", []) or []


# -------------------------------------------------------
# 5) Email derivation helpers (backend-controlled)
#    "Last, First, SKY"  -> "first.last@sky.de"
# -------------------------------------------------------
ORG_SUFFIXES = {"sky", "skyb", "sky de", "sky_de", "sky germany", "sky deutschland"}

def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )

def derive_email(author: str, domain: str = "sky.de") -> str | None:
    """
    Convert 'Last, First, SKY' (Confluence displayName) -> 'first.last@sky.de'
    - Removes org suffix tokens like 'SKY'
    - Handles middle names, hyphens, diacritics
    - Returns None if parsing fails
    """
    if not author or not isinstance(author, str):
        return None

    # Normalize whitespace
    a = " ".join(author.replace("\u00A0", " ").split())
    parts = [p.strip() for p in a.split(",") if p.strip()]

    # Expect "Last, First [, ORG]"
    if len(parts) < 2:
        return None

    last = parts[0]
    first_and_more = parts[1]

    # Remove trailing org token if present in parts[2]
    if len(parts) >= 3:
        org = parts[2].lower().replace(".", "").replace("_", " ")
        if org in ORG_SUFFIXES:
            pass  # ignored
    else:
        # Sometimes "Last, First SKY" (no comma before SKY)
        tokens = first_and_more.split()
        if tokens and tokens[-1].lower() in ORG_SUFFIXES:
            first_and_more = " ".join(tokens[:-1])

    # Keep only alnum/_/- to pick name tokens robustly
    first_tokens = [t for t in re.split(r"[^\w\-]+", first_and_more) if t]
    last_tokens  = [t for t in re.split(r"[^\w\-]+", last) if t]

    if not first_tokens or not last_tokens:
        return None

    first_token = first_tokens[0]
    last_token  = last_tokens[0]

    first_clean = _strip_accents(first_token).lower()
    last_clean  = _strip_accents(last_token).lower()

    if not first_clean or not last_clean:
        return None

    return f"{first_clean}.{last_clean}@{domain}"


def attach_emails(enriched: dict) -> dict:
    """
    If result item is outdated and has an author, attach a mailto link derived as first.last@sky.de.
    """
    if not isinstance(enriched, dict):
        return enriched

    results = enriched.get("results") or []
    if not isinstance(results, list):
        return enriched

    for item in results:
        try:
            if item.get("is_outdated"):
                contact_name = item.get("created_by") or item.get("author")
                if contact_name:
                    email = derive_email(contact_name)
                if email:
                    item["contact_author"] = f"mailto:{email}"
        except Exception:
            # If anything fails for an item, keep going
            continue

    return enriched


# -------------------------------------------------------
# 6) GPT‑4o result-enhancement (final formatting)
# -------------------------------------------------------
def present_results(query: str, results: list[dict]) -> dict:
    """
    Use GPT‑4o to convert raw hybrid DB results into enriched,
    UI-ready JSON with all metadata fields, outdated detection,
    breadcrumb section, cleaned snippets, author info, etc.

    NOTE: The model is instructed NOT to generate email addresses.
          Backend will attach 'contact_author' for outdated items.
    """

    prompt = f"""
You are the AI Search Result Presenter for internal Confluence search.

Your task:
- ONLY use metadata provided below.
- DO NOT hallucinate or invent content.
- Clean and rewrite snippet text (remove URLs/HTML).
- Convert ancestor_titles into a human-readable breadcrumb "section".
- Convert timestamp & created_timestamp into human-readable dates.
- Compute `age_days` = days since last modification (timestamp). If the timestamp is in the future, set age_days = 0.
- Set `is_outdated = true` if age_days > 730.
- Include these fields: author, created_by, timestamp, created_timestamp.
- Do NOT generate email addresses. If the document is outdated, set "email_candidate_hint": true (backend will attach contact_author).
- Return VALID JSON ONLY.

JSON FORMAT TO PRODUCE:
{{
  "summary": "High-level overview of what these documents describe.",
  "results": [
    {{
      "title": "...",
      "url": "...",
      "section": "... > ...",
      "updated": "human-readable date",
      "created": "human-readable date",
      "author": "...",
      "created_by": "...",
      "age_days": 0,
      "is_outdated": false,
      "email_candidate_hint": false,
      "snippet": "rewritten clean snippet",
      "relevance_reason": "why this matches the query"
    }}
  ]
}}

User Query:
{query}

Documents:
{json.dumps(results, indent=2)}
"""

    completion = client_llm.chat.completions.create(
        model=AOAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    raw = completion.choices[0].message.content.strip()

    # ---- Robust code-fence stripping (```json ... ```) ----
    clean = raw
    if clean.startswith("```"):
        # Remove the first fence
        parts = clean.split("```", 1)
        if len(parts) > 1:
            clean = parts[1].strip()
        # Remove optional leading 'json'
        if clean.lower().startswith("json"):
            clean = clean[4:].strip()
    if clean.endswith("```"):
        clean = clean.rsplit("```", 1)[0].strip()

    # ---- Parse JSON and attach derived emails for outdated items ----
    try:
        obj = json.loads(clean)
        obj = attach_emails(obj)
        return obj
    except Exception as e:
        return {
            "error": str(e),
            "raw_response": raw,
            "clean_attempt": clean
        }


# -------------------------------------------------------
# 7) Run this file directly to test
# -------------------------------------------------------
if __name__ == "__main__":
    print("🔍 Fetching 20 random documents from Weaviate...")
    docs = get_random_documents(limit=20)

    if not docs:
        print("❗ No documents retrieved from Weaviate.")
        raise SystemExit()

    print("🤖 Calling GPT‑4o for enhanced result presentation...\n")

    enriched = present_results(
        query="Dummy test query for result formatting",
        results=docs
    )

    print("✅ Enhanced Results:\n")
    try:
        print(json.dumps(enriched, indent=2, ensure_ascii=False))
    except Exception:
        print(enriched)