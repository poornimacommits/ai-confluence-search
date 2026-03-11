import weaviate
import json
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

def to_rfc3339(ms):
    if ms is None:
        return None
    # ms → seconds
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat().replace('+00:00', 'Z')


def resource_name_from_endpoint(endpoint):
    # https://<resource>.openai.azure.com -> <resource>
    if not endpoint:
        return None
    host = endpoint.replace("https://", "").replace("http://", "").split("/")[0]
    return host.split(".")[0] if host else None

def main():
    load_dotenv()
    azure_api_key = (
        os.getenv("AZURE_OPENAI_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENAI_APIKEY")
    )
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment_id = (
        os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        or os.getenv("OPENAI_EMBEDDING_MODEL")
        or "text-embedding-3-small"
    )
    resource_name = os.getenv("AZURE_OPENAI_RESOURCE_NAME") or resource_name_from_endpoint(azure_endpoint)

    if not azure_api_key:
        raise RuntimeError(
            "Missing API key. Set OPENAI_API_KEY, OPENAI_APIKEY, or AZURE_OPENAI_KEY in your environment/.env."
        )
    if not resource_name:
        raise RuntimeError(
            "Missing Azure resource name. Set AZURE_OPENAI_RESOURCE_NAME or AZURE_OPENAI_ENDPOINT in your environment/.env."
        )
    headers = {
        # Required by text2vec-openai at import/query time.
        "X-Openai-Api-Key": azure_api_key,
        "X-Azure-Api-Key": azure_api_key,
    }

    client = weaviate.connect_to_local(headers=headers)

    try:
        # Define your collection using v4-compatible dict config.
        class_dict = {
            "class": "Metadata",
            "properties": [
                {"name": "page_id", "dataType": ["text"]},
                {"name": "title", "dataType": ["text"]},
                {"name": "snippet", "dataType": ["text"]},
                {"name": "version_number", "dataType": ["number"]},
                {"name": "lastmodified_author", "dataType": ["text"]},
                {"name": "created_by", "dataType": ["text"]},
                {"name": "created_timestamp", "dataType": ["date"]}, 
                {"name": "space_key", "dataType": ["text"]},
                {"name": "space_name", "dataType": ["text"]},
                {"name": "ancestor_titles", "dataType": ["text[]"]},
                {"name": "url", "dataType": ["text"]},
                {"name": "lastmodified_timestamp", "dataType": ["date"]},
            ],
            "vectorizer": "text2vec-openai",
            "moduleConfig": {
                "text2vec-openai": {
                    "isAzure": True,
                    "vectorizeClassName": False,
                    "vectorizePropertyName": True,
                    "model": "text-embedding-3-small",
                    "deploymentId": deployment_id,
                    "resourceName": resource_name,
                }
            },
        }

        if client.collections.exists("Metadata"):
            client.collections.delete("Metadata")
            print("Deleted existing 'Metadata' collection.")

        client.collections.create_from_dict(class_dict)
        print("Created 'Metadata' collection.")

        collection = client.collections.get("Metadata")

        with open("confluence_metadata.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        for obj in data:
            obj["created_timestamp"] = to_rfc3339(obj.get("created_timestamp"))
            obj["lastmodified_timestamp"] = to_rfc3339(obj.get("lastmodified_timestamp"))

        collection.data.insert_many(data)
        print(f"Inserted {len(data)} objects")

        # Example semantic search
        search_query = "vector search"
        result = collection.query.near_text(
            query=search_query,
            limit=3,
            return_properties=["title", "snippet"],
        )

        for obj in result.objects:
            print(obj.properties)
    finally:
        client.close()


if __name__ == "__main__":
    main()