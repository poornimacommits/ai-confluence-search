import weaviate
import json
from datetime import datetime, timezone
from weaviate.classes.config import Configure, DataType, Property

# Connect to Weaviate
client = weaviate.connect_to_local()

def to_rfc3339(ms):
    if ms is None:
        return None
    # ms → seconds
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat().replace('+00:00', 'Z')

if client.collections.exists("Metadata"):
    client.collections.delete("Metadata")

# Create a collection
client.collections.create(
    name="Metadata",
    properties=[
    Property(name="page_id", data_type=DataType.TEXT),
    Property(name="title", data_type=DataType.TEXT),
    Property(name="snippet", data_type=DataType.TEXT),
    Property(name="last_modified", data_type=DataType.DATE),
    Property(name="version_number", data_type=DataType.NUMBER),
    Property(name="lastmodified_author", data_type=DataType.TEXT),
    Property(name="space_key", data_type=DataType.TEXT),
    Property(name="space_name", data_type=DataType.TEXT),
    Property(name="url", data_type=DataType.TEXT),
    Property(name="timestamp", data_type=DataType.DATE)
    ],
    vector_config=Configure.Vectors.text2vec_transformers(source_properties=["title", "snippet"])  # Use a vectorizer to generate embeddings during import
    # vector_config=Configure.Vectors.self_provided()  # If you want to import your own pre-generated embeddings
)

# Get collection
meta_data = client.collections.get("Metadata")

with open("confluence_metadata.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for obj in data:
    obj["page_id"] = (obj.get("page_id"))
    obj["title"] = (obj.get("title"))
    obj["snippet"] = (obj.get("snippet"))
    obj["last_modified"] = (obj.get("last_modified")) 
    obj["version_number"] = (obj.get("version_number"))
    obj["lastmodified_author"] = (obj.get("lastmodified_author"))
    obj["space_key"] = (obj.get("space_key"))
    obj["space_name"] = (obj.get("space_name"))
    obj["url"] = (obj.get("url"))
    obj["timestamp"] = to_rfc3339(obj.get("timestamp"))


# Insert data into Weaviate
with meta_data.batch.dynamic() as batch:
    for obj in data:
        batch.add_object(properties=obj)

print(f"Inserted {len(data)} objects")

# Example semantic search
results = meta_data.query.near_text(
    query="vector search",
    limit=3
)

for obj in results.objects:
    print(obj.properties)

client.close()