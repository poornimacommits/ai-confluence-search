import socket
import httpx
from datetime import datetime
from config.settings import settings


# ---------------------------------------------------------
# Proxy & transport helpers
# ---------------------------------------------------------

def vpn_proxy_active(host="127.0.0.1", port=10003, timeout=0.3):
    """Returns True if local VPN proxy is active (home), else False (office)."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except:
        return False


def get_transport():
    """Return proxy transport at home, direct transport in office."""
    if vpn_proxy_active():
        print("Using VPN proxy (127.0.0.1:10003)")
        return httpx.AsyncHTTPTransport(proxy="http://127.0.0.1:10003")
    print("Using direct connection")
    return httpx.AsyncHTTPTransport()


# ---------------------------------------------------------
# Confluence API Client
# ---------------------------------------------------------

class ConfluenceClient:
    def __init__(self):
        self.base = settings.CONFLUENCE_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {settings.CONFLUENCE_TOKEN}",
            "Accept": "application/json",
        }

    # -----------------------------------------------------
    # Basic Confluence API calls
    # -----------------------------------------------------

    async def search(self, query: str):
        """Low-level raw CQL search (useful for debugging)."""
        transport = get_transport()
        params = {
            "cql": f'text ~ "{query}" AND type = page',
            "limit": "5",
            "expand": "content.body.storage,content.version,content.space",
        }

        async with httpx.AsyncClient(transport=transport, timeout=30.0) as client:
            r = await client.get(f"{self.base}/rest/api/search",
                                 headers=self.headers, params=params)
            r.raise_for_status()
            return r.json()

    async def get_page_id(self, space_key: str, title: str):
        """Resolve page ID by space + title."""
        transport = get_transport()
        params = {"spaceKey": space_key, "title": title, "expand": "version,space"}

        async with httpx.AsyncClient(transport=transport, timeout=30.0) as client:
            r = await client.get(f"{self.base}/rest/api/content",
                                 headers=self.headers, params=params)
            r.raise_for_status()
            data = r.json()

        results = data.get("results", [])
        return results[0] if results else None

    async def get_page(self, page_id: str):
        """Fetch full page JSON including storage, version, space, history, ancestors."""
        transport = get_transport()
        params = {"expand": "body.storage,version,space,history,ancestors"}

        async with httpx.AsyncClient(transport=transport, timeout=30.0) as client:
            r = await client.get(f"{self.base}/rest/api/content/{page_id}",
                                 headers=self.headers, params=params)
            r.raise_for_status()
            return r.json()

    async def get_descendants(self, page_id: str, max_nodes=5000):
        """
        BFS traversal to fetch ALL child pages recursively (children, grandchildren, ...).
        Returns a flat list of descendant nodes (NOT full pages).
        """
        queue = [page_id]
        collected = []

        async with httpx.AsyncClient(transport=get_transport(), timeout=30.0) as client:
            while queue and len(collected) < max_nodes:
                parent = queue.pop(0)

                r = await client.get(
                    f"{self.base}/rest/api/content/{parent}/child/page",
                    headers=self.headers,
                    params={"limit": "200", "expand": "version,space"},
                )

                if r.status_code == 404:
                    continue

                r.raise_for_status()
                data = r.json()

                for it in data.get("results", []):
                    collected.append(it)
                    if len(collected) >= max_nodes:
                        break
                    queue.append(it.get("id"))

        return collected

    # -----------------------------------------------------
    # Metadata extraction
    # -----------------------------------------------------

    def extract_metadata(self, page_json: dict, snippet_len: int = 240) -> dict:
        """
        Extract metadata fields for metadata-only DB.
        snippet is taken from the cleaned HTML body for real context.
        """

        content = page_json or {}

        # ---------------------
        # Basic identifiers
        # ---------------------
        page_id = content.get("id")
        title = content.get("title", "")

        # ---------------------
        # Space metadata
        # ---------------------
        space = content.get("space") or {}
        space_key = space.get("key", "")
        space_name = space.get("name", "")

        # ---------------------
        # Ancestor titles (only)
        # ---------------------
        ancestors_raw = content.get("ancestors", []) or []
        ancestor_titles = [a.get("title", "") for a in ancestors_raw]

        # ---------------------
        # Version metadata (last modification)
        # ---------------------
        version = content.get("version") or {}
        last_modified_raw = version.get("when")
        version_number = version.get("number", 0)
        last_author = (version.get("by") or {}).get("displayName", "")

        # numeric last-modified timestamp
        timestamp = None
        if last_modified_raw:
            try:
                dt = datetime.fromisoformat(last_modified_raw.replace("Z", "+00:00"))
                timestamp = int(dt.timestamp() * 1000)
            except:
                timestamp = None

        # ---------------------
        # Creation metadata
        # ---------------------
        history = content.get("history") or {}
        created_by = (history.get("createdBy") or {}).get("displayName", "")
        created_raw = history.get("createdDate")

        created_timestamp = None
        if created_raw:
            try:
                dt2 = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
                created_timestamp = int(dt2.timestamp() * 1000)
            except:
                created_timestamp = None

        # ---------------------
        # URL
        # ---------------------
        links = content.get("_links", {}) or {}
        url = links.get("webui", "")

        # ---------------------
        # Snippet generation
        # ---------------------
        body = (content.get("body") or {}).get("storage") or {}
        html = body.get("value", "") or ""
        clean_text = self._clean_html(html)

        if clean_text.strip():
            snippet = clean_text[:snippet_len].rstrip() + (
                "…" if len(clean_text) > snippet_len else ""
            )
        else:
            snippet = title  # fallback

        # ---------------------
        # Final metadata dict
        # ---------------------
        return {
            "page_id": page_id,
            "title": title,
            "snippet": snippet,
            "version_number": version_number,
            "author": last_author,
            "created_by": created_by,
            "created_timestamp": created_timestamp,
            "space_key": space_key,
            "space_name": space_name,
            "ancestor_titles": ancestor_titles,
            "url": url,
            "timestamp": timestamp,  # last modified epoch ms
        }

    # -----------------------------------------------------
    # High-level metadata pipeline
    # -----------------------------------------------------

    async def fetch_page_with_metadata(self, page_id: str) -> dict:
        """Fetch one page & return metadata-only dict."""
        page_json = await self.get_page(page_id)
        if not page_json:
            return None
        return self.extract_metadata(page_json)

    async def fetch_all_pages_with_metadata(self, root_page_id: str) -> list[dict]:
        """
        1. Get all descendants
        2. For each: fetch full page → extract metadata
        3. Return metadata list ready for indexing
        """
        descendants = await self.get_descendants(root_page_id, max_nodes=5000)

        pages = []
        for node in descendants:
            pid = node.get("id")
            try:
                meta = await self.fetch_page_with_metadata(pid)
                if meta:
                    pages.append(meta)
            except Exception as ex:
                print(f"[warn] Error processing page {pid}: {ex}")
                continue

        return pages

    # -----------------------------------------------------
    # HTML cleaner
    # -----------------------------------------------------

    def _clean_html(self, html: str) -> str:
        """Very small HTML → plain text cleaner for snippet extraction."""
        import re, html as ihtml

        if not html:
            return ""

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", html)

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Unescape HTML entities
        return ihtml.unescape(text)
