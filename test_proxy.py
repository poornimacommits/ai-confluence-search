import asyncio
import httpx
import os
from dotenv import load_dotenv
import socket

load_dotenv()

TOKEN = os.getenv("CONFLUENCE_TOKEN")
BASE_URL = os.getenv("CONFLUENCE_BASE_URL")

def vpn_proxy_active(host="127.0.0.1", port=10003, timeout=0.3):
    try:
        with socket.create_connection((host, port), timeout):
            return True
    except OSError:
        return False

def get_transport():
    if vpn_proxy_active():
        print("Proxy detected: using VPN proxy 127.0.0.1:10003")
        return httpx.AsyncHTTPTransport(proxy="http://127.0.0.1:10003")
    else:
        print("No VPN proxy detected: using direct connection")
        return httpx.AsyncHTTPTransport()

async def main():
    transport = get_transport()

    async with httpx.AsyncClient(transport=transport, timeout=30.0) as client:
        headers = {
            "Authorization": f"Bearer {TOKEN}",
            "Accept": "application/json",
        }
        params = {"cql": 'type = page', "limit": 1}

        r = await client.get(f"{BASE_URL}/rest/api/content/search",
                             headers=headers, params=params)

        print("Status:", r.status_code)
        print("Body:", r.text[:200])

asyncio.run(main())
