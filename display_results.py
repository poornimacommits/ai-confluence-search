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


def fetch(query: str) -> dict:
    load_dotenv()

    AOAI_API_KEY = os.getenv("AZURE_OPENAI_KEY")
    AOAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AOAI_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
    AOAI_MODEL = os.getenv("AZURE_OPENAI_QUERY_REWRITE_DEPLOYMENT")
    # -------------------------------------------------------
    # Azure OpenAI client (GPT‑4o)
    # -------------------------------------------------------
    client_llm = AzureOpenAI(
        azure_endpoint=AOAI_ENDPOINT,
        api_key=AOAI_API_KEY,
        api_version=AOAI_VERSION,
    )

    # -------------------------------------------------------
    # Weaviate v3 client
    # -------------------------------------------------------


    results = get_random_documents(limit=20)
    #result = xyz()

    if not results:
        print("❗ No documents retrieved from Weaviate.")
        raise SystemExit()


    enriched = present_results(
        results,
        query

    )


    return enriched



def get_random_documents(limit=20):
    """
    Retrieve ~random documents by using a random vector search.
    Works with weaviate-client 3.26.6.
    """
    load_dotenv()

    AOAI_API_KEY = os.getenv("AZURE_OPENAI_KEY")
    AOAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AOAI_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
    AOAI_MODEL = os.getenv("AZURE_OPENAI_QUERY_REWRITE_DEPLOYMENT")
    random_vec = np.random.rand(1536).tolist()

    client_wv = weaviate.connect_to_local(
        headers={
            # These headers are used by Weaviate's text2vec-openai module if it needs
            # to call Azure/OpenAI at import/query time.
            "X-Openai-Api-Key": AOAI_API_KEY,
            "X-Azure-Api-Key": AOAI_API_KEY,
        }
    )

    # e.g., "gpt-4o"

    if not AOAI_API_KEY or not AOAI_ENDPOINT or not AOAI_VERSION or not AOAI_MODEL:
        raise RuntimeError(
            "Missing Azure OpenAI ENV vars: AZURE_OPENAI_KEY / AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_VERSION / AZURE_OPENAI_QUERY_REWRITE_DEPLOYMENT")

    collection = client_wv.collections.get("Metadata")

    response = collection.query.near_vector(
        near_vector=random_vec,
        limit=limit,
        return_properties=[
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

    #v4: response.objects is a LIST of objects
    docs = [obj.properties for obj in response.objects]
    client_wv.close()
    return docs or []


def present_results(query: str, results: list[dict]) -> dict:
    """
    Use GPT‑4o to convert raw hybrid DB results into enriched,
    UI-ready JSON with all metadata fields, outdated detection,
    breadcrumb section, cleaned snippets, author info, etc.

    NOTE: The model is instructed NOT to generate email addresses.
          Backend will attach 'contact_author' for outdated items.
    """
    from datetime import datetime

    def serialize_for_prompt(obj):
        """Convert non-JSON types (e.g., datetime) to strings."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        return obj

    system_prompt = f"""You are an assistant that summarizes structured Confluence search results related to stale pages which can be deleted and cleaned.
    You are given:
    1. The user's question
    2. A structured JSON object containing the 10 most relevant Confluence pages
    Your task:
    - Provide a clear, human-readable summary.
    - Highlight outdated pages if applicable.
    - Mention authors when relevant.
    - Explain why the pages match the query.
    - Do NOT invent information.
    
    - Only use the provided JSON.
    - If no strong matches exist, state that clearly."""

    user_prompt = f"""
    User Query:
    {query}

    Top 10 Confluence Results (JSON):
    The schema looks like this:
    
    {json.dumps(results, indent=2, default=serialize_for_prompt)}

    Please provide:
    1. A short, plain-language  summary (2–4 sentences) that answers or frames the user’s query based on the provided pages.
    2. A bullet list of key findings: what each page is about and why it might be relevant and alongside provide a URL link to each page, based on the schema shown above and use the URL parameter which can be found using results[0]['url'] this from results provided and append with https://www.stb.bskyb.com/confluence/ to form the full URL
    4. A list of stale/outdated pages (those with last modified date older than two years, last modified date can be extracted from results[0]['timestamp']), including authors or creators if available.
    5. Practical cleanup suggestions (only if applicable), such as “delete”, “archive”, or “review and update”.
    6. The final answer MUST be written in normal human language — no raw JSON, no code blocks.

    Format the response cleanly using plain text and bullet points.
    """
    load_dotenv()

    AOAI_API_KEY = os.getenv("AZURE_OPENAI_KEY")
    AOAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AOAI_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
    AOAI_MODEL = os.getenv("AZURE_OPENAI_QUERY_REWRITE_DEPLOYMENT")
    client_llm = AzureOpenAI(
        azure_endpoint=AOAI_ENDPOINT,
        api_key=AOAI_API_KEY,
        api_version=AOAI_VERSION,
    )

    completion = client_llm.chat.completions.create(
        model=AOAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
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

# -------------------------------------------------------
# 7) Run this file directly to test
# -------------------------------------------------------
if __name__ == "__main__":
    print("🔍 Fetching 20 random documents from Weaviate...")
    docs = get_random_documents(limit=20)

    if not docs:
        print("❗ No documents retrieved from Weaviate.")
        raise SystemExit()


    enriched = fetch(
        query="Page title related to Suggested Columns in Octane by Fabian",
        results=docs
    )

    print("✅ Enhanced Results:\n")
    try:
        print(json.dumps(enriched, indent=2, ensure_ascii=False))
    except Exception:
        print(enriched)