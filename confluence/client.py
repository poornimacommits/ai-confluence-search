import socket
import httpx
from config.settings import settings


def vpn_proxy_active(host="127.0.0.1", port=10003, timeout=0.3):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except:
        return False


def get_transport():
    if vpn_proxy_active():
        print("Using VPN proxy (127.0.0.1:10003)")
        return httpx.AsyncHTTPTransport(proxy="http://127.0.0.1:10003")
    print("Using direct connection")
    return httpx.AsyncHTTPTransport()


class ConfluenceClient:
    def __init__(self):
        self.base = settings.CONFLUENCE_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {settings.CONFLUENCE_TOKEN}",
            "Accept": "application/json",
        }

    async def search(self, query: str):
        transport = get_transport()
        params = {
            "cql": f'text ~ "{query}" AND type = page',
            "limit": "5",
            "expand": "content.body.storage,content.version,content.space"
        }

        async with httpx.AsyncClient(transport=transport, timeout=30.0) as client:
            r = await client.get(f"{self.base}/rest/api/search",
                                 headers=self.headers, params=params)
            r.raise_for_status()
            return r.json()

    async def get_page_id(self, space_key: str, title: str):
        transport = get_transport()
        params = {"spaceKey": space_key, "title": title, "expand": "version,space"}

        async with httpx.AsyncClient(transport=transport, timeout=30.0) as client:
            r = await client.get(f"{self.base}/rest/api/content",
                                 headers=self.headers, params=params)
            r.raise_for_status()
            data = r.json()

        results = data.get("results", [])
        if not results:
            return None
        return results[0]  # full result block

    async def get_children(self, page_id: str, limit=200, expand="version,space"):
        transport = get_transport()
        base_url = f"{self.base}/rest/api/content/{page_id}/child/page"
        params = {"limit": str(limit), "expand": expand}
        children = []

        async with httpx.AsyncClient(transport=transport, timeout=30.0) as client:
            r = await client.get(base_url, headers=self.headers, params=params)
            r.raise_for_status()
            data = r.json()

        for it in data.get("results", []):
            children.append(it)

        return children

    async def get_page(self, page_id: str):
        transport = get_transport()
        params = {"expand": "body.storage,version,space"}

        async with httpx.AsyncClient(transport=transport, timeout=30.0) as client:
            r = await client.get(f"{self.base}/rest/api/content/{page_id}",
                                 headers=self.headers, params=params)
            r.raise_for_status()
            return r.json()

    async def get_descendants(self, page_id: str, max_nodes=1000):
        """
        BFS traversal of child pages.
        """
        queue = [page_id]
        collected = []

        async with httpx.AsyncClient(transport=get_transport(), timeout=30.0) as client:

            while queue and len(collected) < max_nodes:
                parent = queue.pop(0)

                r = await client.get(
                    f"{self.base}/rest/api/content/{parent}/child/page",
                    headers=self.headers,
                    params={"limit": "200", "expand": "version,space"}
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
