# confluence/build_metadata.py

import asyncio
import json
from datetime import datetime
from confluence.client import ConfluenceClient


async def build_metadata(space_key: str, title: str):
    """
    Fetch metadata for all Confluence pages under the given root page.
    Returns the resulting metadata list so it can be fed into ETL.
    """

    cf = ConfluenceClient()

    # 1) Resolve root page ID
    print(f"[root] Resolving: space='{space_key}', title='{title}'")
    root = await cf.get_page_id(space_key, title)
    if not root:
        raise RuntimeError(f"Root page not found: {space_key} / {title}")

    root_id = root["id"]
    print(f"[root] page_id = {root_id}")

    # 2) All descendants below this root
    print("[desc] Fetching descendants...")
    descendants = await cf.get_descendants(root_id, max_nodes=5000)
    print(f"[desc] Found {len(descendants)} subpages")

    # 3) Loop descendants → fetch metadata
    print("[meta] Fetching metadata for each page...")
    metadata_list = []

    for i, node in enumerate(descendants, 1):
        pid = node.get("id")
        try:
            meta = await cf.fetch_page_with_metadata(pid)
            if meta:
                metadata_list.append(meta)
        except Exception as ex:
            print(f"[warn] Failed to fetch metadata for {pid}: {ex}")

        if i % 20 == 0:
            print(f"[progress] {i}/{len(descendants)} processed")

    print(f"[done] Total metadata records: {len(metadata_list)}")

    return metadata_list


def write_json(records: list[dict], path: str):
    """
    Writes the metadata list to a JSON file.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[file] Metadata written to {path}")


def preview(records: list[dict], n: int = 5):
    """
    Prints the first few entries for inspection.
    """
    print("\n--- SAMPLE OUTPUT (first {0}) ---".format(min(n, len(records))))
    for rec in records[:n]:
        print(rec)
        print()


if __name__ == "__main__":
    # When run directly: build metadata and write to JSON
    out_file = f"metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    records = asyncio.run(
        build_metadata(
            space_key="EE",
            title="End-to-End Test Onbox Team Germany"
        )
    )

    preview(records, n=5)          # show some results
    write_json(records, out_file)  # save entire dataset to JSON