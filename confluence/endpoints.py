from fastapi import APIRouter, Query, HTTPException
from confluence.client import ConfluenceClient

router = APIRouter()
client = ConfluenceClient()


# Search Confluence using CQL (Confluence Query Language) and return raw JSON directly from Confluence.
@router.get("/raw")
async def raw_confluence(q: str):
    return await client.search(q)


# Given a space key and a page title, this endpoint returns the page ID Confluence uses internally.
@router.get("/page_id")
async def api_page_id(space_key: str, title: str):
    result = await client.get_page_id(space_key, title)
    if not result:
        raise HTTPException(404, "Page not found")
    return {
        "page_id": result["id"],
        "title": result["title"],
        "space": result.get("space", {}).get("name")
    }


# fetches one level of Confluence tree
@router.get("/children")
async def api_children(page_id: str):
    children = await client.get_children(page_id)
    return {"page_id": page_id, "children": children}


# fetches ALL levels of Confluence tree
@router.get("/descendants")
async def api_descendants(page_id: str, max_nodes: int = 1000):
    descendants = await client.get_descendants(page_id, max_nodes)
    return {
        "root_page_id": page_id,
        "count": len(descendants),
        "nodes": descendants,
    }


# Download one specific Confluence page
@router.get("/page")
async def api_page(page_id: str):
    page = await client.get_page(page_id)
    if not page:
        raise HTTPException(404, "Page not found")
    return page
